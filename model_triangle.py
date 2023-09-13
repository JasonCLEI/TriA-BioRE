import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss, AFLoss, BCELoss
from triangle import PairStack

class DocREModel_Triangle(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1, num_layers=2):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.num_layers = num_layers
        # self.loss_fnt = ATLoss()
        self.loss_fnt = AFLoss()

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.projection = nn.Linear(emb_size * block_size, config.hidden_size, bias=False)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.pair_stack = PairStack(d_pair=emb_size)
        # self.pair_stack = nn.ModuleList(
        #     [PairStack(d_pair=emb_size) for i in range(num_layers)])

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        b, seq_l, h_size = sequence_output.size()
        n_e = max([len(x) for x in entity_pos])
        hss, tss, rss = [], [], []
        batch_entity_embs = []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            s_ne, _ = entity_embs.size()

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            pad_hs = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
            pad_ts = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
            pad_hs[:s_ne, :s_ne, :] = hs.view(s_ne, s_ne, h_size)
            pad_ts[:s_ne, :s_ne, :] = ts.view(s_ne, s_ne, h_size)

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            m = torch.nn.Threshold(0, 0)
            ht_att = m((h_att * t_att).sum(1))
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-10)

            rs = contract("ld,rl->rd", sequence_output[i], ht_att)

            pad_rs = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
            pad_rs[:s_ne, :s_ne, :] = rs.view(s_ne, s_ne, h_size)

            hss.append(pad_hs)
            tss.append(pad_ts)
            rss.append(pad_rs)

            # hss.append(hs)
            # tss.append(ts)
            # rss.append(rs)

            batch_entity_embs.append(entity_embs)
        hss = torch.stack(hss, dim=0)
        tss = torch.stack(tss, dim=0)
        rss = torch.stack(rss, dim=0)
        # hss = torch.cat(hss, dim=0)
        # tss = torch.cat(tss, dim=0)
        # rss = torch.cat(rss, dim=0)
        batch_entity_embs = torch.cat(batch_entity_embs, dim=0)
        return hss, rss, tss, batch_entity_embs


    def forward(self,
                    input_ids=None,
                    attention_mask=None,
                    labels=None,
                    entity_pos=None,
                    hts=None,
                    instance_mask=None,
                    ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        bs, seq_len, h_size = sequence_output.size()
        bs, num_heads, seq_len, seq_len = attention.size()

        device = sequence_output.device.index

        ne = max([len(x) for x in entity_pos])
        nes = [len(x) for x in entity_pos]

        hs_e, rs_e, ts_e, batch_entity_embs = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs_e = torch.tanh(self.head_extractor(torch.cat([hs_e, rs_e], dim=3)))
        ts_e = torch.tanh(self.tail_extractor(torch.cat([ts_e, rs_e], dim=3)))
        b1_e = hs_e.view(bs, ne, ne, self.emb_size // self.block_size, self.block_size)
        b2_e = ts_e.view(bs, ne, ne, self.emb_size // self.block_size, self.block_size)
        bl_e = (b1_e.unsqueeze(5) * b2_e.unsqueeze(4)).view(bs, ne, ne, self.emb_size * self.block_size)

        # hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        # ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        # b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        # b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        # bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        # logits = self.bilinear(bl)

        feature = self.projection(bl_e)

        feature = self.pair_stack(feature) + feature
        # feature = self.pair_stack(feature)

        # for idx in range(self.num_layers):
        #     feature = self.pair_stack[idx](feature)

        logits_c = self.classifier(feature)

        self_mask = (1 - torch.diag(torch.ones((ne)))).unsqueeze(0).unsqueeze(-1).to(sequence_output)
        logits_classifier = logits_c * self_mask

        logits_classifier = torch.cat([logits_classifier.clone()[x, :nes[x], :nes[x] , :].reshape(-1, self.config.num_labels) for x in range(len(nes))])

        if labels is None:
            logits = logits_classifier.view(-1, self.config.num_labels)
            output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels), logits)

        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(device)

            loss_classifier = self.loss_fnt(logits_classifier.view(-1, self.config.num_labels).float(), labels.float())
            output = loss_classifier

        return output

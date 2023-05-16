import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.drug_encoder.pyg_gnn import PygGNN
from models.text_encoder.xbert import BertConfig, BertForMaskedLM
from models.knowledge_encoder.transe import TransE

from utils.mol_utils import convert_pyg_batch, convert_kge_batch

class MolALBEF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_n_nodes = config["max_n_nodes"]
        bert_config = BertConfig.from_json_file(config["bert_config_path"])

        self.graph_encoder = PygGNN(
            num_layer=config["gin_num_layers"],
            emb_dim=config["gin_hidden_dim"],
            gnn_type="gin",
            drop_ratio=config["drop_ratio"],
            JK="last",
        )
        if "gin_ckpt" in config:
            self.graph_encoder.load_state_dict(torch.load(config["gin_ckpt"]))
        self.graph_proj_head = nn.Linear(config["gin_hidden_dim"], config["projection_dim"])
        self.graph_linear = nn.Linear(config["gin_hidden_dim"], bert_config.hidden_size)

        self.text_encoder = BertForMaskedLM(bert_config)
        if "bert_ckpt" in config:
            ckpt = torch.load(config["bert_ckpt"])
            processed_ckpt = {}
            if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in ckpt:
                for k, v in ckpt.items():
                    if k.startswith("module.ptmodel."):
                        processed_ckpt[k[15:]] = v
                    else:
                        processed_ckpt[k] = v
            missing_keys, unexpected_keys = self.text_encoder.load_state_dict(processed_ckpt, strict=False)
            #print("missing keys:", missing_keys)
            #print("unexpected keys:", unexpected_keys)
        self.text_proj_head = nn.Linear(bert_config.hidden_size, config["projection_dim"])
        
        if "kge" in config:
            self.kg_encoder = TransE(**config["kge"])
            self.kg_proj_head = nn.Linear(config["kge"]["hidden_size"], config["projection_dim"])
            self.kg_linear = nn.Linear(config["kge"]["hidden_size"], bert_config.hidden_size)

        self.mtm_head = nn.Linear(bert_config.hidden_size, 2)

        self.temperature = 0.1
        self.device = "cuda:0"
        # TODO: 
        self.text_linear = nn.Linear(bert_config.hidden_size, config["projection_dim"])
        self.gin_hidden_dim = config["gin_hidden_dim"]
        self.output_dim = config["gin_hidden_dim"] + config["projection_dim"]
        # for test
        self.output_dim = config["gin_hidden_dim"] + bert_config.hidden_size
        #self.output_dim = config["gin_hidden_dim"]
        

    def forward(self, mol, text, kg=None, cal_loss=False):
        mol_embeds, node_embeds = self.graph_encoder(mol)
        # TODO: mol_feats?
        mol_feats = F.normalize(self.graph_proj_head(mol_embeds), dim=-1)
        all_node_feats = self.graph_linear(node_embeds)
        # serialize node feature
        batch_size = mol_feats.shape[0]

        node_feats, node_attention_mask = convert_pyg_batch(all_node_feats, mol.batch, self.max_n_nodes)

        text_outputs = self.text_encoder.bert(text["input_ids"], attention_mask=text["attention_mask"], mode='text', return_dict=True)
        seq_feats = text_outputs["last_hidden_state"]
        
        if kg is not None:
            # TODO:
            # neigh_feats = self.kg_encoder.predict(kg["neigh_indice"])
            
            all_neigh_feats = []
            for i in kg:
                neigh_feat = self.kg_encoder.predict(i)
                neigh_feat = self.kg_linear(neigh_feat)
                all_neigh_feats.append(neigh_feat)
            neigh_feats, neigh_attention_mask = convert_kge_batch(all_neigh_feats)
            # TODO:dropout text and kg embedding
            node_feats = torch.cat((node_feats, neigh_feats), dim=1)
            node_attention_mask = torch.cat((node_attention_mask, neigh_attention_mask), dim=1)
        
        output = self.text_encoder.bert(
            encoder_embeds=seq_feats,
            attention_mask=text["attention_mask"],
            encoder_hidden_states=node_feats,
            encoder_attention_mask=node_attention_mask,
            mode='fusion',
            return_dict=True
        )
        if cal_loss:
            perm = []
            for i in range(batch_size):
                j = i
                while j == i:
                    j = random.randint(0, batch_size - 1)
                perm.append(j)
            perm = torch.LongTensor(perm).to(seq_feats.device)
            output_neg = self.text_encoder.bert(
                encoder_embeds=seq_feats,
                attention_mask=text["attention_mask"],
                encoder_hidden_states=node_feats[perm],
                encoder_attention_mask=node_attention_mask[perm],
                mode='fusion',
                return_dict=True
            )
            label = torch.cat((torch.ones(batch_size), torch.zeros(batch_size)), dim=0).long().to(seq_feats.device)
            logits = self.mtm_head(torch.cat((output["last_hidden_state"][:, 0, :], output_neg["last_hidden_state"][:, 0, :]), dim=0))
            return F.cross_entropy(logits, label)
        else:
            return output, mol_embeds

    def encode_structure(self, structure, proj=True, return_node_feats=False):
        drug_embeds, node_embeds = self.graph_encoder(structure)
        if proj:
            drug_embeds = self.graph_proj_head(drug_embeds)
        if not return_node_feats:
            return drug_embeds
        else:
            return drug_embeds, node_embeds
          
    def encode_structure_with_all(self, mol, text, kg=None, cal_loss=False):
        output, drug_embeds = self.forward(mol, text, kg=None, cal_loss=False)
        text_embeds = F.dropout(output["last_hidden_state"][:,0,:], 0.5, training=self.training)
        # text_embeds = self.text_linear(text_embeds)
        # drug_embeds = self.graph_proj_head(drug_embeds)
        return torch.cat((drug_embeds, text_embeds), dim=1)
         
    def encode_structure_with_kg(self, mol, kg, proj=True):
        # get structure embedding first 
        drug_embeds, _ = self.graph_encoder(mol)
        kg_feats = self.kg_encoder.predict(kg)
        if proj:
            drug_embeds = self.graph_proj_head(drug_embeds)
            kg_feats = self.kg_proj_head(kg_feats)
        feats = torch.cat((drug_embeds, kg_feats.squeeze()), dim=1)
        return feats
    
    def encode_all_module(self, mol, kg, text, proj=True):
        # get structure embedding first 
        # TODO:
        self.output_dim = 256 * 3
        drug_embeds, _ = self.graph_encoder(mol)
        kg_feats = []
        for i in kg:
            kg_feat = self.kg_encoder.predict(i).mean(0)
            kg_feats.append(kg_feat)
        kg_feats = torch.cat([i.unsqueeze(0) for i in kg_feats], 0)
        text_embeds = self.text_encoder.bert(text["input_ids"], attention_mask=text["attention_mask"], mode='text', return_dict=True)["last_hidden_state"]
        if proj:
            drug_embeds = self.graph_proj_head(drug_embeds)
            kg_feats = self.kg_proj_head(kg_feats)
            text_embeds = self.text_proj_head(text_embeds)
        feats = torch.cat((drug_embeds, kg_feats, text_embeds[:, 0, :].squeeze()), dim=1)
        return feats
    
    
    def encode_structure_with_prob(self, structure, x, atomic_num_list, device):
        drug_embeds, _ = self.graph_encoder(structure, x, atomic_num_list, device)
        return self.graph_proj_head(drug_embeds) 

    def encode_text(self, text, return_cls=True, proj=True):
        text_embeds = self.text_encoder.bert(text["input_ids"], attention_mask=text["attention_mask"], mode='text', return_dict=True)["last_hidden_state"]
        if return_cls:
            text_embeds = text_embeds[:, 0, :]
        if proj:
            return self.text_proj_head(text_embeds)
        else:
            return text_embeds

    def encode_knowledge(self, kg):
        return self.kg_encoder.predict(kg)

    def predict_similarity_score(self, data):
        preds = self.forward(data["structure"], data["text"])["last_hidden_state"][:, 0, :]
        return F.softmax(self.mtm_head(preds), dim=-1)[:, 1]

    def calculate_matching_loss(self, drug, text):
        return self.forward(drug, text, cal_loss=True)
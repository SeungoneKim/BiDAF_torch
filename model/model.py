import torch
import torch.nn as nn
from model.emb import CharEmb, WordEmb, ContextualEmb
from model.layer import HighwayNetwork, AttnFlow, Modeling, Output

class BiDAF(nn.Module):
    def __init__(BiDAF, self):
        # Embedding Query to acquire Query Representation
        self.CharEmb_Query = CharEmb()
        self.WordEmb_Query = WordEmb()
        self.HighwayNetwork_Query = HighwayNetwork()
        self.ContextualEmb_Query = ContextualEmb()
        
        # Embedding Context to acquire Context Representation
        self.CharEmb_Cont = CharEmb()
        self.WordEmb_Cont = WordEmb()
        self.HighwayNetwork_Cont = HighwayNetwork()
        self.ContextualEmb_Cont = ContextualEmb()

        # Modeling Output with Query Representation, Context Representation
        self.AttnFlowLayer = AttnFlow()
        self.ModelingLayer = Modeling()
        self.OutputLayer = Output()
    
    
    # query length : J / context length : T
    # hs1 == hs2 == 300
    def __forward__(self, query, context):
        q1 = self.CharEmb(query)     # (bs,hs1,sl1)
        q2 = self.WordEmb(query)     # (bs,hs2,sl1)
        c1 = self.CharEmb(context)   # (bs,hs1,sl2)
        c2 = self.WordEmb(context)   # (bs,hs2,sl2)

        q = self.HighwayNetwork_Query(torch.cat(q1,q2,dim=0)) # (bs,hs,sl1), hs=hs1+hs2
        c = self.HighwayNetwork_Cont(torch.cat(c1,c2,dim=0))  # (bs,hs,sl2), hs=hs1+hs2
        
        queryAwareRepresentation = self.AttnFlowLayer(q,c)    # (bs,hs*4,sl2)
        output = self.ModelingLayer(queryAwareRepresentation) # (bs,hs,sl2)
        output = self.OutputLayer(output)                     # TASK SPECIFIC

        return output

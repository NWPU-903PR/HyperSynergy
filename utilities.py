# +
class Mapping():  
    def __init__(self, items):        
        self.item2idx={}
        self.idx2item=[]
        
        for idx, item in enumerate(items):
            self.item2idx[item]=idx
            self.idx2item.append(item)
            
    def add(self,item):
        if item not in self.idx2item:
            self.idx2item.append(item)
            self.item2idx[item]=len(self.idx2item)-1




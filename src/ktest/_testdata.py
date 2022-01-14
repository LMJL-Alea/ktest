

class TestData:
    
    from .initializations import init_data

    def __init__(self,x,y,x_index=None,y_index=None,variables=None,kernel=None):
        self.init_data(x,y,x_index,y_index,variables,kernel=None)
        
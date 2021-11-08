

class TestData:
    
    from .initializations import init_data

    def __init__(self,x,y,kernel=None,x_index=None,y_index=None,variables=None):
        self.init_data(x,y,kernel,x_index,y_index,variables)
        
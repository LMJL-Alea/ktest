
from .kernels import gauss_kernel,linear_kernel,gauss_kernel_mediane,fisher_zero_inflated_gaussian_kernel,gauss_kernel_weighted_variables,gauss_kernel_mediane_per_variable





def get_kernel_name(function,bandwidth,median_coef):
    n = ''
    if function in ['gauss','fisher_zero_inflated_gaussian']:
        n+=function
        if bandwidth == 'median':
            n+= f'_{median_coef}median' if median_coef != 1 else '_median' 
        else: 
            n+=f'_{bandwidth}'
    elif function == 'linear':
        n+=function
    elif function == 'gauss_kernel_mediane_per_variable':
        n+=function
    else:
        n='user_specified'
    return(n)

def init_kernel_params(function='gauss',
                       bandwidth='median',
                       median_coef=1,
                       kernel_name=None,
                       ):
    """
    Returns a dict containing the parameters that specify the kernel function to compute.
    
    Parameters : 
    ------------
        function (default = 'gauss') : function or str in ['gauss','linear','fisher_zero_inflated_gaussian','gauss_kernel_mediane_per_variable'] 
            if str : specifies the kernel function
            if function : kernel function specified by user

        bandwidth (default = 'median') : 'median' or float
            value of the bandwidth for kernels using a bandwidth
            if 'median' : the bandwidth will be set as the median or a multiple of it is
                according to the value of parameter `median_coef`
            if float : value of the bandwidth

        median_coef (default = 1) : float
            multiple of the median to use as bandwidth if bandwidth=='median' 

        kernel name (default = None) : str
            The name of the kernel function specified by the call of the function.
            if None : the kernel name is automatically generated through the function
            get_kernel_name 


    """
    return(
        {'function':function,
            'bandwidth':bandwidth,
            'median_coef':median_coef,
            'kernel_name':kernel_name,
            }
    )




class Kernel_Function:

    def init_kernel(self,
               function='gauss',
               bandwidth='median',
               median_coef=1,
               kernel_name=None,
               verbose=0):
        '''
        
       Parameters : 
    ------------
        function (default = 'gauss') : function or str in ['gauss','linear','fisher_zero_inflated_gaussian','gauss_kernel_mediane_per_variable'] 
            if str : specifies the kernel function
            if function : kernel function specified by user

        bandwidth (default = 'median') : 'median' or float
            value of the bandwidth for kernels using a bandwidth
            if 'median' : the bandwidth will be set as the median or a multiple of it is
                according to the value of parameter `median_coef`
            if float : value of the bandwidth

        median_coef (default = 1) : float
            multiple of the median to use as bandwidth if bandwidth=='median' 
            
        kernel name (default = None) : str
            The name of the kernel function specified by the call of the function.
            if None : the kernel name is automatically generated through the function
            get_kernel_name 

        Returns
        ------- 
        '''

        data = self.get_data(in_dict=False)
        has_bandwidth = False
        kernel_name = get_kernel_name(function=function,bandwidth=bandwidth,median_coef=median_coef) if kernel_name is None else kernel_name
        
        if function == 'gauss':
            has_bandwidth = True
            kernel_,computed_bandwidth = gauss_kernel_mediane(x=data,y=None,
                                                bandwidth=bandwidth,  
                                               median_coef=median_coef,
                                               return_mediane=True,
                                               verbose=verbose)

        elif function == 'linear':
            kernel_ = linear_kernel

        else:
            kernel_ = function

        self.data[self.data_name]['kernel'] = kernel_
        self.data[self.data_name]['kernel_name'] = kernel_name
        if has_bandwidth:
            self.data[self.data_name]['kernel_bandwidth'] = computed_bandwidth
        self.has_kernel = True
        self.kernel_params = init_kernel_params(function=function,
                                                bandwidth=bandwidth,
                                                median_coef=median_coef,
                                                kernel_name=kernel_name,
                                                )

    def get_kernel_params(self):
        return(self.kernel_params.copy())

    def get_kernel(self):
        kernel = self.data[self.data_name]['kernel']
        return(kernel)
    
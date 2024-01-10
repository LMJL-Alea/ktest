
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
                       weights=None,
                       weights_power=1,
                       kernel_name=None,
                       pi1=None,
                       pi2=None):
    """
    Returns an object that defines the kernel
    """
    return(
        {'function':function,
            'bandwidth':bandwidth,
            'median_coef':median_coef,
            'kernel_name':kernel_name,
            'weights':weights,
            'weights_power':weights_power,
            'pi1':pi1,
            'pi2':pi2
            }
    )




class Kernel_Function:

    def init_kernel(self,
                function='gauss',
               bandwidth='median',
               median_coef=1,
               kernel_name=None,
               weights = None,
               weights_power = 1,
               pi1=None,
               pi2=None,
               all_data=False,
               verbose=0):
        '''
        
        Parameters
        ----------
            function (default = 'gauss') : str or function
                str in ['gauss','linear','fisher_zero_inflated_gaussian','gauss_kernel_mediane_per_variable'] for gauss kernel or linear kernel. 
                function : kernel function specified by user

            bandwidth (default = 'median') : str or float
                str in ['median'] to use the median or a multiple of it as a bandwidth. 
                float : value of the bandwidth

            median_coef (default = 1) : float
                multiple of the median to use as bandwidth if kernel == 'gauss' and bandwidth == 'median' 

            pi1,pi2 (default = None) : None or str 
                if function == 'fisher_zero_inflated_gaussian' : columns of the metadata containing 
                the proportions of zero for the two samples   

            all_data:
                Set the kernel for generalized hypothesis testing.
                (to do) I did not allow to specify the kernel in this case 
        Returns
        ------- 
        '''

        if all_data:
            data = self.get_data(samples='all',in_dict=False)
            kernel,mediane = gauss_kernel_mediane(data,None,return_mediane=True)
            self.kernel_all_data = kernel

        if verbose >0:
            s=f'- Define kernel function'
            if verbose ==1:
                s+=f' ({function})'
            else:
                s+=f'\n\tfunction : {function}'
                if function == 'gauss':
                    s+=f'\n\tbandwidth : {bandwidth}'
                if bandwidth == 'median' and median_coef != 1:
                    s+=f'\n\tmedian_coef : {median_coef}'
                if kernel_name is not None:
                    s+=f'\n\tkernel_name : {kernel_name}'
            print(s)
        
        x,y = self.get_xy()
        has_bandwidth = False

        kernel_name = get_kernel_name(function=function,bandwidth=bandwidth,median_coef=median_coef) if kernel_name is None else kernel_name
        if verbose>1:
            print("kernel_name:",kernel_name)
        if function == 'gauss':
            has_bandwidth = True
            if weights is not None:
                if isinstance(weights,str):
                    if weights in self.get_var():
                        weights_ = self.get_var()[weights]
                    elif weights in ['median','variance']:
                        weights_=weights
                    else: 
                        print(f"kernel weights '{weights}' not recognized.")
                else:
                    weights_ = weights
                kernel_,computed_bandwidth = gauss_kernel_weighted_variables(x=x,y=y,
                                                                           weights=weights_,
                                                                           weights_power=weights_power,
                                                                           bandwidth=bandwidth,
                                                                          median_coef=median_coef,
                                                                          return_mediane=True,
                                                                          verbose=verbose)

            else:
                kernel_,computed_bandwidth = gauss_kernel_mediane(x=x,y=y,      
                                                bandwidth=bandwidth,  
                                               median_coef=median_coef,
                                               return_mediane=True,
                                               verbose=verbose)



        elif function == 'linear':
            kernel_ = linear_kernel
        elif function == 'fisher_zero_inflated_gaussian':
            has_bandwidth = True
            kernel_,computed_bandwidth = fisher_zero_inflated_gaussian_kernel(x=x,y=y,
                                                                    pi1=pi1,pi2=pi2,
                                                                    bandwidth=bandwidth,
                                                                    median_coef=median_coef,
                                                                    return_mediane=True,
                                                                    verbose=verbose)
        # elif function == 'gauss_kernel_mediane_per_variable':
        #     has_bandwidth = True
        #     kernel_,computed_bandwidth = gauss_kernel_mediane_per_variable(x=x,y=y,
        #                                                                    bandwidth=bandwidth,
        #                                                                   median_coef=median_coef,
        #                                                                   return_mediane=True,
        #                                                                   verbose=verbose)



        else:
            kernel_ = function


        if verbose>1:
            print("kernel",kernel_)
        self.data[self.data_name]['kernel'] = kernel_
        self.data[self.data_name]['kernel_name'] = kernel_name
        if has_bandwidth:
            self.data[self.data_name]['kernel_bandwidth'] = computed_bandwidth
        self.has_kernel = True
        self.kernel_params = init_kernel_params(function=function,
                                                bandwidth=bandwidth,
                                                median_coef=median_coef,
                                                weights=weights,
                                                weights_power=weights_power,
                                                kernel_name=kernel_name,
                                                pi1=pi1,pi2=pi2)

    def get_kernel_params(self):
        return(self.kernel_params.copy())

    def get_kernel(self,all_data=False):
        if all_data:
            kernel=self.kernel_all_data
        else:
            kernel = self.data[self.data_name]['kernel']
        return(kernel)
    
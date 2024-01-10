




class Names:
                 
    # Names 

    def get_data_to_test_str(self,condition=None,samples=None,marked_obs_to_ignore=None):

        dn = self.data_name
        c,samples,mark = self.init_samples_condition_marked(condition=condition,
                                           samples=samples,
                                           marked_obs_to_ignore=marked_obs_to_ignore)

        # si les conditions et samples peuvent être mis en entrées, cente_by aussi
        smpl = '' if samples == 'all' else "".join(samples)
        cb = '' if self.center_by is None else f'_cb_{self.center_by}'    
        marking = '' if mark is None else f'_{mark}'
        return(f'{dn}{c}{smpl}{cb}{marking}')

    def get_model_str(self):
        ny = self.nystrom
        nys = 'ny' if self.nystrom else 'standard'
        ab = f'_basis{self.anchor_basis}' if ny else ''
        lm = f'_lm{self.landmark_method}_m{self.get_n_landmarks()}' if ny else ''
        return(f'{nys}{lm}{ab}')

    def get_landmarks_name(self):
        dtn = self.get_data_to_test_str()


        lm = self.landmark_method
        n_landmarks = f'_m{self.get_n_landmarks()}'
        
        return(f'lm{lm}{n_landmarks}_{dtn}')

    def get_kmeans_landmarks_name_for_sample(self,sample):
        landmarks_name = self.get_landmarks_name()
        return(f'{sample}_{landmarks_name}')

    def get_anchors_name(self,):
        dtn = self.get_data_to_test_str()

        lm = self.landmark_method
        ab = self.anchor_basis
        n_landmarks = f'_m{self.get_n_landmarks()}'
        # r = f'_r{self.r}' # je ne le mets pas car il change en fonction des abérations du spectre
        return(f'lm{lm}{n_landmarks}_basis{ab}_{dtn}')

    def get_covw_spev_name(self):
        dtn = self.get_data_to_test_str()
        mn = self.get_model_str()

        return(f'{mn}_{dtn}')

    def get_kfdat_name(self,condition=None,samples=None,marked_obs_to_ignore=None):
        dtn = self.get_data_to_test_str(condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
        mn = self.get_model_str()

        return(f'{mn}_{dtn}')

    def get_orthogonal_name(self,t,center,condition=None,samples=None,marked_obs_to_ignore=None):
        dtn = self.get_data_to_test_str(condition=condition,samples=samples,marked_obs_to_ignore=marked_obs_to_ignore)
        
        mn = self.get_model_str()

        c = center
        return(f'{c}{t}_{mn}_{dtn}')
        
    def get_mmd_name(self):
        dtn = self.get_data_to_test_str()
        mn = self.get_model_str()
        return(f'{mn}_{dtn}')

    def get_corr_name(self,proj):
        if proj in ['proj_kfda','proj_kpca']:
            name = f"{proj.split(sep='_')[1]}_{self.get_kfdat_name()}"
        else : 
            print(f'the correlation with {proj} is not handled yet.')
        return(name)

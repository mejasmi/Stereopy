from typing import Union, Sequence
from copy import deepcopy
import anndata
import numpy as np
import scvi

from stereo.algorithm.algorithm_base import AlgorithmBase
from stereo.core.stereo_exp_data import StereoExpData, AnnBasedStereoExpData
from stereo.io.reader import stereo_to_anndata
from stereo.log_manager import LogManager
from stereo.preprocess.filter import filter_genes


class TotalVi(AlgorithmBase):

    def main(
        self,
        protein_data: Union[StereoExpData, AnnBasedStereoExpData],
        protein_list: Sequence[str] = None,
        use_highly_genes: bool = True,
        hvg_res_key: str = 'highly_variable_genes',
        res_key: str = 'totalVI'
    ):
        assert self.stereo_exp_data.shape == protein_data.shape

        if self.stereo_exp_data.raw is None:
            raise Exception("there is no raw data, please run data.tl.raw_checkpoint before normalization.")
        
        assert self.stereo_exp_data.shape == self.stereo_exp_data.raw.shape

        if use_highly_genes:
            data = self._sub_by_hvg(hvg_res_key)
        else:
            data = self.stereo_exp_data

        if isinstance(rna_data, StereoExpData):
            LogManager.stop_logging()
            rna_data: anndata.AnnData = stereo_to_anndata(data, split_batches=False)
            LogManager.start_logging()
        else:
            rna_data: anndata.AnnData = deepcopy(data._ann_data)
        
        rna_data.layers['count'] = deepcopy(data.raw.exp_matrix)
        if protein_list is not None:
            protein_data = self._sub_protein_by_name(protein_data, protein_list)
        
        rna_data.obsm['protein_exp_matrix'] = deepcopy(protein_data.exp_matrix)
        rna_data.uns['protein_names'] = deepcopy(protein_data.genes.gene_name)

        scvi.model.TOTALVI.setup_anndata(
            rna_data,
            protein_expression_obsm_key='protein_exp_matrix',
            protein_names_uns_key='protein_names',
            layer='counts'
        )

        total_vi = scvi.model.TOTALVI(rna_data)
        total_vi.train()
        self.pipeline_res[res_key] = total_vi.get_latent_representation()


    
    def _sub_by_hvg(self, hvg_res_key='highly_variable_genes'):
        assert hvg_res_key in self.pipeline_res

        highly_variable = self.pipeline_res[hvg_res_key]['highly_variable']
        data = deepcopy(self.stereo_exp_data)
        data.sub_by_index(gene_index=highly_variable)
        data.raw.sub_by_index(gene_index=highly_variable)
        return data
    
    def _sub_protein_by_name(
        self,
        protein_data: Union[StereoExpData, AnnBasedStereoExpData],
        protein_list
    ):
        gene_index = np.isin(protein_data.genes.gene_name, protein_list)
        return protein_data.sub_by_index(gene_index=gene_index)

    # def _create_anndata(self):
    #     adata = anndata.AnnData(
    #         X=self.stereo_exp_data.exp_matrix
    #     )
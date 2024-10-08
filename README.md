# scRGCL
ScRGCL
# Requirements
- python = 3.6.7
- pytorch = 1.1.0
- pytorch-geometric = 1.3.1
- sklearn
# Installation
Download scRGCL by
```
git clone https://github.com/nathanyl/scRGCL
```
# Instructions
## Preprocessing data for model training
```
python gen_data.py <options> -expr <expr_mat_file> -label <expr_label_file> -net <network_backbone_file>  -out <outputfile>
```
```
Arguments:
  expr_mat_file: scRNA-seq expression matrix with genes as rows and cells as columns (csv format)
  e.g.   EntrezID,barocode1,barocode2,barocode3,barocode3
          5685,1,0,0,0
          5692,0,0,0,0
          6193,0,0,0,1

  expr_label_file: cell types assignments (csv format)
  e.g. Barcodes ,label
        barocode1, celltype1
        barocode2, celltype1
        barocode3, celltype2
        barocode4, celltype3
  
  network_backbone_file: gene interactin network backbone (csv format)
  e.g. STRING database,1~3 cloumns indicate gene1, gene2 and combined_score respectively. Genes are in Entrez ID format.
      23521,6193,999
      5692,5685,999
      5591,2547,999
      6222,25873,999
  
  outputfile: preprocessed data for model training (npz format)
 
Options:
  -q <float> the top q quantile of network edges are used (default: 0.99 for STRING database)
```
## Run scRGCL model
```
python scRGCL.py -in <inputfile> -out-dir <outputfolder> -bs <batch_size>
```
```
 Arguments:  
  inputfile: preprocessed data for model training (npz format)  
  outputfolder: the folder in which prediction results are saved 
  batch_size : batch size for model training
```

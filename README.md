# scRGCL
ScRGCL is a GNN-based automatic cell identification algorithm leveraging gene interaction relationships to enhance the performance of the cell type identification.
# Requirements
- python = 3.6.7
- pytorch = 1.1.0
- pytorch-geometric = 1.3.1
- sklearn
# Installation
Download scGraph by
```
git clone https://github.com/nathanyl/scRGCL
```
# Instructions
## Preprocessing data for model training
```
python gen_data.py <options> -expr <expr_mat_file> -label <expr_label_file> -net <network_backbone_file>  -out <outputfile>
```
## Run scGraph model
```
python scRGCL.py -in <inputfile> -out-dir <outputfolder> -bs <batch_size>
```

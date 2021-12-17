# find_phone

I've used an R-CNN implementation to solve the task at hand.

## Setup
There are 2 ways to run the project-
1. Using conda
To setup the environemtn using conda, one can simply run -
`conda env create -f conda_env.yml`

2. On Collab
To setup on collab, one can follow the following steps (also shown in Vaibhav_Mathur_find_phone.ipynb) -
    - Clone repo 'git clone https://github.com/vaibhav117/find_phone.git'
    - Change directory to folder find_phone
    - run `sh setup.sh` 

## Potentail issues
1. I've trained the model on CPU as the particular implementation of R-CNN used is quite old. This leads to a long training duration.
# JZ templates

The purpose of this repo is to provide templates for people who don't have direct access to JZ but who have been working on experiments that we want to run on JZ.

## How to design your experience for JZ

To designate an experiment on JZ you have to think in stages:

1. Uploading stage: stage where we have to download from the internet everything we need (dataset, model, tokenizer, dependencies, etc);
2. CPU stage: stage in which only CPU operations are performed (typically pre-processing);
3. GPU stage: stage during which only operations on the CPU and GPU are performed. This is typically training and evaluation;
4. Downloading stage: stage during which the outputs (checkpoints, datasets, metrics) are retrieved from JZ.

What I propose is to put on the Hub the data that will be uploaded to JZ (dataset, initial model, tokenizer, etc).

Concretely, to work on JZ, you have to prepare bash scripts (and more precisely SLURM files) which will be put in a job queue to be executed. You will find in the `experiments/jz/templates/SLURM/experiment_template` folder templates of scripts to realize an end-2-end experiment. Each of these scripts are composed of 2 sections:

1. A section indicating the characteristics of the job to run (typically its duration and the hardware to use);
2. A section which is a bash script in which you just have to list the terminal commands to run to launch a part of the experiment.

You will also find in `experiments/jz/templates/SLURM/experiment_example` folder an example of an experiment that could be launched on JZ.

As you will certainly not be able to run these scripts yourself on JZ, what I suggest is that you write the bash instructions to be used for your experiments (keeping in mind the need to think of your experiment in steps with one script per type of step). Don't hesitate to write your doubts or questions while writing this script so that we can discuss them before the execution of the script on the cluster.

As a tip, try to prepare a toy example to check that your scripts can be run on JZ. By toy example I mean a small enough dataset that we can run the jobs with very little time and compute. Indeed, as the jobs are put in a queue there is a priority system which makes that the small jobs are executed more quickly. If ever there is a small bug in the code it can be very useful to be able to debug it quickly.

In summary some interesting points to know about JZ:

-   the computational partitions **do not have access to the internet**. We use specific partions for everything that needs the internet.
-   we try to use only **2 conda environments**: a stable one which corresponds to the dependencies on master and a development one. If your experiment requires dependencies that are not on the master branch of our repository you will have to tell the person who will run your experiment
-   we have several storage partitions on JZ, if your code uses **paths** you will also have to talk to the person who will launch your experiment. For your information:
    1. The dataset clones are located at `$DATASETS_CUSTOM`
    2. The clone of the repo on the master branch is at `$WORK/repos/metadata/`
    3. The wandb logs are at `$SCRATCH/wandb` (deleted after 30 days if there is no access to the file in the meantime)
    4. The checkpoints are located at `$SCRATCH/metadata_outputs/{job_id}` (deleted after 30 days if the file has not been accessed in the meantime)
    5. For scripts requiring GPU computing we try to use one 16GB V100 (maximum 20h).
    6. For scripts requiring CPU computation we try to use a maximum of 1 node (40 CPUs).

If you ever get stuck on anything to design your experience on JZ, contact me. The instructions will most likely change according to your needs.

It might be interesting to plan some peers coding sessions to prepare experiments that would go beyond this very generic framework. But in any case, it will be useful to have a bash script base to visualize the operations to perform.

## Downloading from JZ

This is not yet ready:

-   downloading the checkpoints (do we want to send them to the HUB?)
-   logging in tensorboard to be able to use [this feature](https://huggingface.co/bigscience/tr3d-1B3-more-warmup-tensorboard/tensorboard) of the HUB

What is ready:

-   synchronization of wandb logs (on request)

## Checklist

Here is a small checklist of information that the person running your script will probably want to know:

-   What do you need to download (dataset, template, tokenizer)?
-   Where are your scripts located in the repository, in what order should they be run?
-   Are you using the master branch of modelling-metadata? If not, why not?
-   Do your dependencies match the dependencies listed on master?

**COMP3710** Getting Started on Rangpur 

Semester 2, 2026 

#### **C O M P 3 7 1 0 • S E M E S T E R 2 , 2 0 2 6** 

# **Getting Started on Rangpur** 

HPC quick-start guide 

The University of Queensland 

This guide gets you onto **Rangpur** , UQ’s HPC cluster, and walks you through installing PyTorch and submitting your first GPU job: one that reports whether CUDA is available. Read Section 1 first, since understanding _how_ an HPC works will save you a lot of confusion later. For anything this guide does not cover, ask your tutors, and you are encouraged to read more widely about Slurm and HPC workflows on your own. 

## **1 HPC basics: what you are actually using** 

An HPC (High Performance Computing) cluster is not one computer. It is many computers ( **nodes** ) sharing a network and a common filesystem, coordinated by a **scheduler** . The key ideas: 

- **Login node.** When you SSH in you land on a shared login node ( `login` ). It is for editing files, moving data, and submitting jobs, _not_ for heavy computation. Everyone shares it, so never run training or long scripts here. 

- **Compute nodes.** The machines that do real work (the A100 GPU nodes, CPU nodes, etc.). You never SSH into them directly; you _ask the scheduler_ for time on them. 

- **Scheduler (Slurm).** Rangpur uses **Slurm** . You describe the resources you need (a GPU, some memory, a time limit) and Slurm queues your request, then runs it on a free compute node. 

- **Partitions (queues).** Compute nodes are grouped into partitions. This course uses the `comp3710` partition (A100 GPU nodes) for GPU work, and a `cpu` partition for CPU-only tasks. 

- **Shared filesystem.** Your home directory is visible from the login node _and_ every compute node, so a file you create on the login node is available inside your jobs. 

**Note** The golden rule of shared HPC: **the login node is a lobby, not a workshop.** Do your computation inside a Slurm job, never directly on `login` . 

There are two ways to run work through Slurm: 

- **Interactive job** ( `srun ... --pty bash` ): gives you a live shell on a compute node. Good for setup, debugging, and quick tests. 

- **Batch job** ( `sbatch job.sh` ): submits a script that runs unattended. This is how you should run real workloads. It queues, runs, and writes output to a file even after you log off. 

## **2 Connecting to Rangpur** 

From a terminal (macOS/Linux, or Windows PowerShell / WSL), SSH in with your UQ student login. Replace `sXXXXXXX` with your own student number everywhere in this guide. 

1 

**COMP3710** Getting Started on Rangpur 

Semester 2, 2026 

```
sshsXXXXXXX@rangpur.compute.eait.uq.edu.au
```

**Note** You must be on the UQ network or connected to the **UQ VPN** for this to work off-campus. Enter your UQ password when prompted. 

Once in, you are on the login node and your home directory is: 

```
/home/Student/sXXXXXXX#thisisalsothevalueof$HOME
```

**Watch out** Everything you do lives in your home directory ( `$HOME` ), which has a storage quota. Avoid keeping large datasets or old conda environments you no longer need. 

You can inspect the cluster’s partitions and node states with: 

```
sinfo
```

Look for the `comp3710` rows. Those are the A100 GPU nodes reserved for this course, and the partition you will request for GPU jobs. 

## **3 Installing Miniconda + PyTorch** 

You will install Miniconda into your home directory and create a `torch` environment. **Do this inside an interactive job** , not on the login node. Installing needs _no GPU_ , so use a **CPU node** and keep the scarce A100s free for jobs that actually train. 

### **3.1 Step 1: Grab an interactive session on a CPU node** 

```
srun--partition=cpu--time=01:00:00--ptybash
```

Your prompt will change once Slurm places you on a compute node. Now run the following commands **one at a time** , reading each before you paste it. 

### **3.2 Step 2: Point temp files at your home directory** 

```
exportTMPDIR=$HOME/tmp
exportTMP=$HOME/tmp
exportTEMP=$HOME/tmp
mkdir-p$HOME/tmp
```

2 

**COMP3710** Getting Started on Rangpur 

Semester 2, 2026 

### **3.3 Step 3: Download the Miniconda installer** 

```
cd$HOME
curl-Ohttps://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

### **3.4 Step 4: Run the installer (unattended, into $HOME/miniconda3)** 

```
bashMiniconda3-latest-Linux-x86_64.sh-b-p$HOME/miniconda3
```

The `-b` flag runs a batch (no prompts) install; `-p` sets the install location. 

### **3.5 Step 5: Activate conda** 

```
source$HOME/miniconda3/bin/activate
```

### **3.6 Step 6: Accept the Anaconda channel Terms of Service** 

Recent conda versions require this once before creating environments. 

```
condatosaccept--override-channels--channelhttps://repo.anaconda.com/pkgs/main
condatosaccept--override-channels--channelhttps://repo.anaconda.com/pkgs/r
```

### **3.7 Step 7: Create the** `torch` **environment** 

```
condacreate-ntorchpython=3.11pip-y
```

### **3.8 Step 8: Activate it and install PyTorch** 

```
condaactivatetorch
pip3install--no-cache-dirtorchtorchvision
```

`–no-cache-dir` avoids filling your home quota with pip’s download cache. 

### **3.9 Step 9: Check the install worked** 

You are on a CPU node, so don’t expect CUDA here. Just confirm PyTorch imports cleanly: 

```
python-c"importtorch;print('torch',torch.__version__,'installedOK')"
```

You’ll confirm that `CUDA is available` in the next section, from a job that actually runs on a GPU. 

3 

**COMP3710** Getting Started on Rangpur 

Semester 2, 2026 

**Note** To make conda available automatically in future logins, run `conda init bash` once, then log out and back in. After that, `conda activate torch` works straight from the login node. 

### **3.10 Step 10: Release the interactive session** 

When setup is done, free the node for others: 

```
exit
```

## **4 Your first batch job: does CUDA work?** 

Now the real objective: submit a job that runs on a GPU node and reports CUDA availability. Two small files, both created in your home directory. 

### **4.1 The Python script** 

Create `test_cuda.py` : 

```
importtorch
print("CUDAavailable:",torch.cuda.is_available())
iftorch.cuda.is_available():
print("Devicecount:",torch.cuda.device_count())
print("Devicename:",torch.cuda.get_device_name(0))
```

### **4.2 The Slurm submission script** 

Create `job.sh` . Every line starting with `#SBATCH` is a directive read by the scheduler _before_ your job runs. 

```
#!/bin/bash
#SBATCH--job-name=cuda-check
#SBATCH--partition=comp3710
#SBATCH--gres=gpu:1
#SBATCH--mem=16G
#SBATCH--time=00:05:00
#SBATCH--output=cuda_check_%j.out
#SBATCH--error=cuda_check_%j.err
#Activateyourcondaenvironment
source$HOME/miniconda3/bin/activate
condaactivatetorch
#Runthescript
pythontest_cuda.py
```

What each directive means: 

- `–job-name` : a label you’ll see in the queue. 

4 

**COMP3710** Getting Started on Rangpur 

Semester 2, 2026 

- `–partition=comp3710` : run on the course’s A100 nodes. 

- `–gres=gpu:1` : request one GPU (without this, `torch.cuda.is_available()` is `False` ). 

- `–mem=16G` : memory reserved for the job. 

- `–time=00:05:00` : max wall time ( `HH:MM:SS` ); the job is killed if it runs over. 

- `–output` / `–error` : where stdout/stderr go; `%j` is replaced by the job ID. 

### **4.3 Submit and monitor** 

```
sbatchjob.sh#submit;printsajobID
squeue--me#seeyourqueued/runningjobs
scancel<jobid>#cancelajobifneeded
```

When the job finishes, read its output: 

```
catcuda_check_<jobid>.out
```

You should see `CUDA available: True` along with the A100 device name. edit on the login node _→_ queue with Slurm _→_ run on a GPU node _→_ collect output. Congratulations, you’re doing HPC. 

## **5 HPC good practice** 

- **Never compute on the login node.** Editing, `git` , and small file ops are fine. Training, big installs, and data crunching always go inside a Slurm job. 

- **Request only what you need.** Over-requesting GPUs, memory, or time makes you wait longer in the queue and blocks classmates. Under-requesting gets your job killed. Tune it. 

- **Use CPU nodes for CPU work.** Installs, data prep, and light scripting don’t need a GPU, so save the A100s for training. 

- **Use short time limits while debugging.** A 5-minute job schedules far faster than a 10-hour one. Grow the limit only once the code works. 

- **Mind your home quota.** Everything you keep counts against it, so clean out old environments, installers ( `Miniconda3-latest-*.sh` ), caches, and checkpoints you no longer need. 

- **Prefer batch jobs over interactive ones** for anything slow, since they don’t die when your laptop sleeps or your SSH connection drops. 

- **Free interactive sessions when done** ( `exit` ). A held GPU you aren’t using is a GPU nobody else can use. 

- **Check before you scale.** Run once on a tiny input and short time limit to confirm it works, then submit the full job. 

## **6 Slurm script good practice** 

- **Always set** `–job-name` **,** `–time` **,** `–mem` **, and** `–partition` explicitly. Don’t rely on defaults you haven’t checked. 

5 

**COMP3710** Getting Started on Rangpur 

Semester 2, 2026 

- **Route output with** 

   - `%j` (e.g. `run_%j.out` ) so repeated submissions don’t overwrite each other’s logs. 

- **Activate your environment inside the script** , not just in your interactive shell, because a batch job starts from a clean environment. 

- **Echo useful context** at the top of the script for debugging, e.g.: 

```
echo"Job$SLURM_JOB_IDon$(hostname),started$(date)"
nvidia-smi#confirmtheGPUyouweregiven
```

- **Match your request to your code.** If your script uses one GPU, request one GPU. Asking for more it can’t use just wastes the allocation. 

- **Keep scripts in version control** , parameterised where possible, so a job is reproducible and easy to resubmit. 

## **Command cheat sheet** 

|**Command**|**What it does**|
|---|---|
|`ssh sXXXXXXX@rangpur...uq.edu.au`|Connect to the login node|
|`sinfo`|List partitions and node states|
|`srun --partition=cpu --pty bash`|Interactive CPU session (setup / light work)|
|`srun --partition=comp3710`<br>`--gres=gpu:1 --pty bash`|Interactive GPU session (debugging)|
|`sbatch job.sh`|Submit a batch job|
|`squeue --me`|Show your jobs in the queue|
|`scancel <jobid>`|Cancel a job|
|`conda activate torch`|Activate your PyTorch environment|
|`exit`|Leave an interactive session / log out|



**Note** Everywhere you see `sXXXXXXX` , substitute your own UQ student number. Your home directory is `/home/Student/sXXXXXXX` , which `$HOME` expands to automatically. 

6 


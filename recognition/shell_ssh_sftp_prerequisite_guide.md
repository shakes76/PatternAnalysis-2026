# **COMP3710 Prerequisite Guide: Shell, SSH Remote Access, and SFTP File Transfer** 

Welcome to COMP3710! In this course, you will build, train, and evaluate deep learning models —including Convolutional Neural Networks (CNNs) and Transformers—on GPU hardware. 

Before diving into PyTorch and GPU execution, you need a solid foundation in three key technical skills: 

##### 1. **Using a Command-Line Shell Environment** 

##### 2. **Connecting to Remote Servers via SSH** 

3. **Transferring Files between Local and Remote Systems via SFTP** 

This guide provides a comprehensive, beginner-friendly introduction to these concepts across **Windows** , **macOS** , and **Linux** desktop environments. It will prepare you to follow the course GPU guides in the Ed thread Setting Up Your GPU Environment. More resources can be found under Ed Resources. 

## **Table of Contents** 

1. <u>Module 1: Shell Environment Fundamentals</u> 

2. <u>Module 2: Platform-Specific Shell Environment Setup</u> 

3. <u>Module 3: SSH (Secure Shell) Remote Access</u> 

4. Module 4: SFTP & File Transfer Workflows 

5. Module 5: Troubleshooting FAQ & Next Steps 

## **Module 1: Shell Environment Fundamentals** 

A **shell** (or terminal) is a text-based interface used to control your computer by typing commands instead of clicking with a mouse. High-performance computing (HPC) clusters and cloud servers do not have graphical desktops, so operating through a shell is an essential skill. 

### **1.1 Opening the Terminal on Your OS** 

##### **Windows** : 

Search for **Windows Terminal** or **PowerShell** in the Start Menu. 

_Recommended_ : Use **Windows Terminal** (or launch Ubuntu via WSL after setup). 

##### **macOS** : 

Press `Cmd + Space` to open Spotlight, type `Terminal` , and press `Enter` (or use `iTerm2` if installed). 

##### **Linux** : 

Press `Ctrl + Alt + T` or search for `Terminal` in your application menu. 

### **1.2 Essential Navigation Commands** 

In Unix/Linux shells (and macOS/WSL), commands follow the structure: `command [options] [arguments]` . 

|**Command**|**Action**|**Example**<br>**Usage**|**Description**|
|---|---|---|---|
|`pwd`|**P**rint**W**orking<br>**D**irectory|`pwd`|Displays the absolute path of your current<br>folder.|
|`ls`|**L**i**s**t files|`ls -la`|Lists files.<br>`-l`shows detailed stats;<br>`-a`<br>shows hidden files.|
|`cd`|**C**hange**D**irectory|`cd project`|Moves into the<br>`project`directory.|
|`cd ..`|Move up|`cd ..`|Moves up one parent directory level.|
|`cd ~`or<br>`cd`|Go Home|`cd ~`|Returns to your home directory (<br>`$HOME` ).|
|`mkdir`|**M**ake**D**irectory|`mkdir`<br>`comp3710`|Creates a new directory named<br>`comp3710` .|



### **1.3 File Viewing and Manipulation** 

|**Command**|**Action**|**Example Usage**|**Description**|
|---|---|---|---|
|`cp`|**C**o**p**y file/folder|`cp train.py`<br>`train_backup.py`|Copies<br>`train.py` . Add<br>`-r`to copy a<br>folder recursively.|
|`mv`|**M**o**v**e / Rename|`mv old.py new.py`|Moves or renames a file or<br>directory.|
|`rm`|**R**e**m**ove file|`rm temp.py`|Deletes a file permanently (**no**<br>**trash bin!**). Use<br>`-r`for folders.|
|`cat`|Concatenate /<br>Display|`cat script.py`|Prints the full content of a text file to<br>the terminal screen.|
|`head`/<br>`tail`|View top /<br>bottom|`head -n 20 log.txt`|Displays the first (or last) 20 lines of<br>a file.|
|`nano`|Terminal Text<br>Editor|`nano script.py`|Simple terminal file editor (<br>`Ctrl+O`<br>save,<br>`Ctrl+X`exit).|



### **1.4 Directory Paths & Environment Variables** 

**Home Directory (** **`~` or** **`$HOME` )** : 

On Linux/Rangpur: `/home/Student/sXXXXXXX` 

On WSL: `/home/yourusername` 

On macOS: `/Users/yourusername` 

##### **Path Separators** : 

Unix/Linux/macOS use forward slashes: `/home/user/project` 

Windows uses backslashes: `C:\Users\user\project` 

##### **Current vs Parent Directory** : 

- `.` refers to the current directory (e.g., `python ./script.py` ). 

- `..` refers to the parent directory (e.g., `cd ..` ). 

### **1.5 Productivity Tips** 

- **Tab Autocompletion** : Type the first few letters of a command or file path and press `Tab` to auto-complete it. Press `Tab` twice to list all matches. 

- **Command History** : Use the `Up` and `Down` arrow keys to cycle through previously executed commands. 

- **Cancel Running Command** : Press `Ctrl + C` in the terminal to terminate a running script or command immediately. 

**Clear Screen** : Type `clear` or press `Ctrl + L` to clean up your terminal screen. 

[!TIP] **Module 1 Practice Checkpoint** : Open a terminal on your computer and complete the following steps: 

1. `cd` into you home directory `~` 

2. Run `pwd` to see where you are. 

3. Create a folder named `comp3710_test` : `mkdir comp3710_test` 

4. Move into it: `cd comp3710_test` 

5. Create a file using nano: `nano hello.py` -> Type `print("Hello COMP3710")` , press `Ctrl+O` then `Enter` to save, `Ctrl+X` to exit. 

6. Verify the file exists and view its contents: `ls -l` followed by `cat hello.py` . 

## **Module 2: Platform-Specific Shell Environment Setup** 

### **2.1 Windows Users: Understanding PowerShell vs. WSL** 

On Windows, you have two shell environments: 

1. **PowerShell** : The default Windows command shell. Useful for launching SSH sessions ( `ssh sXXXXXXX@rangpur...` ) or opening WSL. 

2. **WSL (Windows Subsystem for Linux)** : A full Ubuntu Linux environment running natively inside Windows. 

[!IMPORTANT] **WSL File Performance Tip** : In WSL, your Windows `C:` drive is mounted under `/mnt/c/` . While you can access `/mnt/c/` , working on files inside 

> `/mnt/c/` from Linux tools is **significantly slower** . **Always keep your code and Conda environments inside your Linux home directory (** **`~` or** **`$HOME` )** . 

To start Ubuntu inside Windows Terminal / PowerShell: 

```
wsl
```

### **2.2 macOS Users** 

macOS runs a Unix kernel natively, and the default shell is **Zsh** . You can run standard Linux commands ( `ls` , `cd` , `mkdir` , `ssh` , `scp` ) directly inside the macOS Terminal app without needing WSL or virtual machines. 

### **2.3 Linux Users** 

Linux users can open any standard terminal (Bash/Zsh) to run local commands, manage Python environments, and connect to remote servers directly. 

## **Module 3: SSH (Secure Shell) Remote Access** 

### **3.1 What is SSH and Why Do We Use It?** 

**SSH (Secure Shell)** is a cryptographic network protocol that allows you to connect securely to a remote computer (like UQ's Rangpur HPC cluster) over an encrypted connection. 

When you SSH into Rangpur, your terminal window acts as a window into the remote machine: commands you type are executed on Rangpur's hardware. 

#### **Architectural Mental Model: Local Laptop vs. HPC Cluster** 



<!-- Start of picture text -->
+-------------------+        SSH        +-----------------------------------+<br>|   Your Laptop     | =================>| Rangpur Login Node (shared lobby) |<br>| (Windows/Mac/Lin) |                   |  - Editing files, small git ops   |<br>+-------------------+                   |  - Submitting Slurm jobs          |<br>                                        +-----------------------------------+<br>                                                          |<br>                                                    Slurm Scheduler<br>                                                          |<br>                                                          v<br>                                        +-----------------------------------+<br>                                        |  Slurm Compute Node (A100 GPU)    |<br>                                        |  - Heavy training & model execution|<br>                                        +-----------------------------------+<br><!-- End of picture text -->

[!IMPORTANT] **Golden Rule of Shared HPC** : The login node is a shared lobby, not a workshop! Never run heavy model training on the login node. Always use Slurm ( `srun` or `sbatch` ) to request a compute node. 

### **3.2 Connecting to Rangpur HPC** 

To connect to Rangpur, open your local terminal (Windows PowerShell/WSL, macOS Terminal, or Linux Terminal) and run: 

```
ssh sXXXXXXX@rangpur.compute.eait.uq.edu.au
```

_(Replace_ _`sXXXXXXX` with your actual UQ student number)._ 

[!WARNING] **Network Requirements (UQ VPN)** : You must be connected to the **UQ Campus Wi-Fi (UQNAVIGATOR / eduroam)** or connected to the **UQ VPN** when off-campus for SSH connections to Rangpur to succeed. 

#### **First-Time Connection Prompt** 

The first time you connect, you will see a fingerprint verification prompt: 

```
The authenticity of host 'rangpur.compute.eait.uq.edu.au (...)' can't be established.
ED25519 key fingerprint is SHA256:...
```

```
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Type `yes` and press `Enter` . 

#### **Password Prompt** 

Enter your UQ student password when prompted. _(Note: As a security feature, the cursor will_ **_not move or show asterisks_** _while typing your password. Simply type your password and press_ _`Enter` )._ 

### **3.3 Disconnecting from SSH** 

To end your SSH session and return to your local desktop prompt, type: 

```
exit
```

### **3.4 Key-Based Authentication (Passwordless SSH Setup)** 

Instead of typing your password every time you SSH into Rangpur, you can set up SSH keys: 

##### 1. **Generate an SSH key pair on your local desktop** : 

```
ssh-keygen -t ed25519 -C"student_number@uq.edu.au"
```

Press `Enter` to accept the default file location and optional passphrase. 

##### 2. **Copy your public key to Rangpur** : 

##### **macOS / Linux / WSL** : 

```
ssh-copy-id sXXXXXXX@rangpur.compute.eait.uq.edu.au
```

##### **Windows PowerShell** : 

```
type$env:USERPROFILE\.ssh\id_ed25519.pub | ssh
sXXXXXXX@rangpur.compute.eait.uq.edu.au "cat >> ~/.ssh/authorized_keys"
```

Now, typing `ssh sXXXXXXX@rangpur.compute.eait.uq.edu.au` will log you in automatically! 

### **3.5 VS Code Remote - SSH (Recommended IDE Setup)** 

Visual Studio Code provides an extension called **Remote - SSH** that lets you edit files, open terminal sessions, and run code directly on Rangpur or inside WSL with a graphical IDE experience. 

1. Install **Visual Studio Code** on your local laptop/desktop. 

2. Open VS Code, go to the Extensions view ( `Ctrl+Shift+X` or `Cmd+Shift+X` ), and search for **Remote - SSH** (by Microsoft). Click **Install** . 

3. Press `F1` (or `Cmd+Shift+P` on macOS), type `Remote-SSH: Connect to Host...` , and enter: 

```
sXXXXXXX@rangpur.compute.eait.uq.edu.au
```

4. A new VS Code window will open connected to Rangpur. You can browse remote files, edit Python scripts, and launch terminals directly on the remote server. 

5. Please note that you should not run any code using VSCode right now, as you are still on a login node through a different interface. Always run scripts using Slurm batch files, as detailed below. 

##### [!TIP] **Module 3 Practice Checkpoint** : 

1. Open your terminal and connect to Rangpur: `ssh sXXXXXXX@rangpur.compute.eait.uq.edu.au` 

2. Verify your home directory location by typing `pwd` (should show `/home/Student/sXXXXXXX` ). 

3. Disconnect by typing `exit` . 

## **Module 4: SFTP & File Transfer Workflows** 

When training deep learning models, you will often need to move dataset files, code scripts, or model checkpoints ( `.pt` / `.pth` files) between your local computer and the remote Rangpur cluster. **SFTP (SSH File Transfer Protocol)** handles secure file transfers over SSH. 

### **4.1 GUI-Based SFTP Clients (Beginner-Friendly)** 

GUI clients provide a drag-and-drop dual-pane interface: your local computer's files on the left, and Rangpur's remote files on the right. 

#### **Option A: WinSCP (Windows Only)** 

**WinSCP** is a popular SFTP client for Windows. 

##### 1. **Download & Install** : Free download from winscp.net. 

##### 2. **Configure Connection** : 

   - **File protocol** : `SFTP` 

   - **Host name** : `rangpur.compute.eait.uq.edu.au` 

   - **Port number** : `22` 

   - **User name** : `sXXXXXXX` (your student ID) 

   - **Password** : Your UQ password (or leave blank if using SSH key) 

3. Click **Save** (e.g. name it "Rangpur HPC"), then click **Login** . 

##### 4. **Transferring Files** : 

Left side = Local Windows files. 

- Right side = Remote Rangpur home directory ( `/home/Student/sXXXXXXX` ). 

- Drag and drop files or folders between the panes to upload or download. 

#### **Option B: FileZilla (Windows, macOS, Linux)** 

**FileZilla** is a cross-platform SFTP client. 

##### 1. **Download & Install** : Download FileZilla Client from filezilla-project.org. 

##### 2. **Configure Connection via Site Manager** : 

- Go to `File` -> `Site Manager` -> `New Site` . 

- **Protocol** : `SFTP - SSH File Transfer Protocol` 

- **Host** : `rangpur.compute.eait.uq.edu.au` 

- **Port** : `22` 

- **Logon Type** : `Normal` (or `Key file` if using SSH keys) 

- **User** : `sXXXXXXX` 

- **Password** : Your UQ password 

##### 3. Click **Connect** . 

4. **Transferring Files** : Drag and drop files between the local tree (left) and remote tree (right). 

#### **Option C: Cyberduck (Windows & macOS)** 

**Cyberduck** is an intuitive file transfer client for Windows and macOS. 

1. Download from cyberduck.io. 

2. Click **Open Connection** . 

3. Select **SFTP (SSH File Transfer Protocol)** from the dropdown. 

4. Server: `rangpur.compute.eait.uq.edu.au` , Port: `22` , Username: `sXXXXXXX` , Password: UQ password. 

5. Click **Connect** and drag files into the window to upload. 

### **4.2 Command-Line File Transfer Tools** 

If you prefer working entirely in the terminal, these CLI tools enable fast transfers: 

#### **1.** **`scp` (Secure Copy)** 

Run `scp` from your **local terminal** (do not run it inside the remote Rangpur session!). 

##### **General Syntax** : 

```
scp[options][user@]host1:source_file [user@]host2:destination_file
```

##### **Upload a file to Rangpur** : 

```
scp local_script.py sXXXXXXX@rangpur.compute.eait.uq.edu.au:~/
```

##### **Download a file from Rangpur** : 

```
scp sXXXXXXX@rangpur.compute.eait.uq.edu.au:~/output.log ./
```

##### **Upload a whole folder recursively (** **`-r` )** : 

```
scp-r ./my_dataset sXXXXXXX@rangpur.compute.eait.uq.edu.au:~/datasets/
```

#### **2. Interactive** **`sftp` Prompt** 

Start an interactive transfer session from your local terminal: 

```
sftp sXXXXXXX@rangpur.compute.eait.uq.edu.au
```

Once connected to the `sftp>` prompt: 

- `ls` / `cd` – List / change directory on **remote** server. 

- `lls` / `lcd` – List / change directory on **local** machine. 

- `put local_file.py` – Upload file to remote server. 

- `get remote_file.py` – Download file to local machine. 

- `quit` – Exit SFTP prompt. 

#### **3.** **`rsync` (Fast & Efficient Sync)** 

`rsync` syncs directories by transferring only modified files. Ideal for large datasets or model checkpoints. 

```
rsync-avz-P ./my_code/ sXXXXXXX@rangpur.compute.eait.uq.edu.au:~/my_code/
```

_(Flags:_ _`-a` archive mode,_ _`-v` verbose,_ _`-z` compress,_ _`-P` show progress)._ 

### **4.3 Drag-and-Drop via VS Code** 

If you are connected to Rangpur using the **VS Code Remote - SSH** extension (described in Section 3.5), you can transfer files simply by dragging them from your local computer's File Explorer / Finder directly into the VS Code file explorer sidebar! 

##### [!TIP] **Module 4 Practice Checkpoint** : 

1. Use your preferred SFTP method (WinSCP, FileZilla, Cyberduck, or `scp` ) to upload your `hello.py` file created in Module 1 to your Rangpur home directory. 

2. SSH into Rangpur and verify `hello.py` is present in your `$HOME` directory. 

## **Module 5: Troubleshooting FAQ & Next Steps** 

### **5.1 Common Troubleshooting FAQ** 

|**Problem / Error**<br>**Message**|**Probable Cause**|**Recommended Solution**|
|---|---|---|
|`ssh: connect to host`<br>`... Connection timed`<br>`out`|You are off-campus<br>without UQ VPN<br>active, or server is<br>down.|Connect to the UQ VPN client, or ensure<br>you are connected to<br>`eduroam`/<br>`UQNAVIGATOR`Wi-Fi on campus.|
|`Permission denied`<br>`(publickey,password)`|Incorrect password or<br>wrong student ID<br>(<br>`sXXXXXXX` ).|Re-type password carefully. Ensure<br>username starts with<br>`s`followed by 7<br>digits.|
|Cursor does not move<br>when typing password|Expected Linux<br>security behavior.|Keep typing your password blindly and<br>press<br>`Enter` .|
|`Host key verification`<br>`failed`|Server IP or host key<br>changed.|Remove the old key entry:<br>`ssh-keygen -R`<br>`rangpur.compute.eait.uq.edu.au`and<br>retry.|
|`Disk quota exceeded`or<br>`No space left`|`$HOME`storage quota<br>full from Conda<br>caches/datasets.|Clean up installers (<br>`rm Miniconda3-*.sh` ),<br>clear pip cache (<br>`pip cache purge` ), or<br>delete unused environments.|
|Slow file performance in<br>Windows WSL|Working inside<br>`/mnt/c/`Windows<br>drive.|Move your project folder into Linux home:<br>`cp -r /mnt/c/Users/you/project ~/`and<br>work from<br>`~/` .|



### **5.2 Quick Readiness Self-Check** 

- **Shell** : Can you navigate directories using `cd` , list files with `ls -la` , create folders with `mkdir` , and view files with `cat` / `nano` ? 

- **SSH** : Can you SSH into `rangpur.compute.eait.uq.edu.au` using your UQ student ID while on campus or via UQ VPN? 

- **SFTP** : Do you have an SFTP method ready (WinSCP, FileZilla, Cyberduck, `scp` , or VS Code Remote) to transfer files? 

### **5.3 Choose Your GPU Development Workflow** 

Depending on whether you are working in the university labs or remotely on the HPC, open the corresponding setup guide in the Ed thread Setting Up Your GPU Environment: 

##### 1. **Working on University Lab Computers (Local WSL + RTX 2080 Super GPU)** : 

   - Read and follow: COMP3710_lab_computers.pdf 

   - _Key takeaway_ : You will launch `wsl` in Windows, set up Miniconda + PyTorch in Linux `$HOME` , and run scripts locally on the RTX 2080 Super GPU. 

2. **Working on UQ's Rangpur HPC Cluster (Remote Slurm + NVIDIA A100 GPUs)** : 

   - Read and follow: COMP3710-Rangpur.pdf 

- _Key takeaway_ : You will SSH into Rangpur login node, request an interactive CPU session ( `srun` ), install Miniconda + PyTorch, and submit training jobs ( `sbatch` ) to run on A100 GPU nodes. 

Good luck with your deep learning experiments in COMP3710! 


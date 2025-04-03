## Installing Visual Studio Code

#### [VS Code Website](https://code.visualstudio.com/Download)

## Command Line Basics

#### Echo prints in the terminal whatever parameter we pass it.

```bash
echo Rage Against the Machine Learning
```

#### pwd stands for print working directory and it prints the "place" or directory we are currently at in the computer.

```bash
pwd
```

#### ls presents you the contents of the directory you're currently in. It will present you with both the files and other directories your current directory contains.

```bash
ls
```

#### cd is short for Change directory and it will take you from your current directory to another. You can also write cd + space + drag your desired folder to move to that location.

```bash
cd Desktop
```

#### mkdir stands for make directory and it will create a new directory for you. You have to pass the command the directory name parameter.

```bash
mkdir myfolder
```

#### rmdir stands for Remove directory 

```bash
rmdir myfolder
```

#### touch allows you to create an empty file in your current directory. As parameters it takes the file name.
MacOS only:
```bash
touch myfile.txt
```
On Windows Powershell do this instead:
```bash
ni myfile.txt
```
Or on Command Prompt:
```bash
echo > myfile.txt
```

#### to remove the file:

```bash
rm myfile.txt
```

#### You can clone files from github to your folder. 
There might be an error that git is not installed. 
On MacOS it should prompt you to install developer tools, please install them. 
For Windows, please follow this [Link](https://git-scm.com/downloads/win), and install the package.
Then re-run the command.

```bash
git clone https://github.com/fdoblhammer/Machine-Learning-for-Artists.git
```

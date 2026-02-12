# Colony Counter

## Context

In microbiology, to estimate bacterial or viruses growth on plates, it is needed to count the colonies by hand.
However, the manual process is:

- slow and repetitive
- difficult to scale when many plates must be processed
- error-prone 
- when wells are hard to count, different operators may disagree on the result

This project provides an **automatic colony counting** tool for 12-well plates, using machine learning.
It includes:

- a **preprocessing** to allow the user to put the full picture without worrying to crop the 12 wells
- a **countability classifier** that detects whether a well is countable,  
- a **regression model (counter)** that predicts the number of colonies in countable wells,

By applying the same counting model to all images, the tool provides **consistent estimates**, reducing subjective differences between plates.

## Design of the Model

The system is a **two layer pipeline**, so it works in 2 main steps inspired by how a human operator would approach the task:

### **1. Countability Classification**
Some wells are too dense to be counted correctly and can be defined as uncountable.  
The first model predicts:

- **0 = uncountable**
- **1 = countable**

### **2. Colony Counting (Regression Model)**
If a well is classified as countable, a second neural network estimates the **number of colonies** it contains.

Multiple models are supported and all are registered and managed in `MODEL_DICTIONARY`.  

### **3. Output format**

The pipeline is designed to process **many images at once**.

- You can provide **multiple image files**
- Images can be organized in **one or several folders**
- All images are processed automatically in a single run

After processing, **all results are saved in a single Excel file** (`.xlsx`).

---

## 0. Installation and Setup

### 1) Install Conda (only if you don’t have it)

Install **Miniconda** from the official website depending on your OS (Windows/Mac/Linux):  
https://www.anaconda.com/download/success

Do not check anything not recommended.
Anaconda works too if you have it installed already but Miniconda is lighter.
This will allow you to use Python and its libraries safely.

After installation, check that Conda is installed by running this line in a terminal :

```bash
conda --version
```
If a version number is displayed, Conda is correctly installed.

If it is not working, go to the troubleshooting section further below.

### 2) Download the project

You need to download the project files to your computer before installing or running the program.
There are two possible ways to do this. Choose the one that fits your situation.

#### A) If you have Git (recommended)

Git is a tool often used by developers to download and update projects.

1. Open a terminal.
2. Go to the folder where you want to store the project
3. Download the project by running:
```bash
git clone https://github.com/CS-433/project-2-wells_colony_counter.git
```
4. Enter the project folder: ```cd project-2-wells_colony_counter```

You are now inside the **project root folder**.
This is the main folder that contains files such as `requirements.txt` and the `usage` folder.

#### B) If you do NOT have Git (simplest method)

This method does not require any technical tools.

1. Open your web browser.
2. Go to the GitHub page of the project : https://github.com/CS-433/project-2-wells_colony_counter
3. Click the green **Code** button.
4. Select **Download ZIP**.
5. Extract the ZIP file to a folder on your computer (for example, Documents).

After extraction, open the project folder.
You should see files such as `requirements.txt` and a folder named `usage`.

Once the project is downloaded and you are inside the project root folder,
you can continue with the installation steps.

### 3) Create the Conda environment

From the **root of the repository(project)**, create a new Conda environment with Python by opening a terminal writing these command lines :

```bash
conda create -n colony python=3.10
conda activate colony
```
It may ask you to agree terms which you can agree by typing `a` and then accept to proceed by typing `y`.
This creates a separate “bubble” on your computer that contains everything needed for this project, without affecting anything else on your system.

If it is not working, go to the troubleshooting section further below.
### 4) Install project dependencies

Install all required libraries using pip (still from the root of the repository):

```bash
pip install -r requirements.txt
```
It may take some time since it need to download all libraries needed.
This installs all scientific, image-processing, and machine-learning libraries needed to run the project.

When installation is completed, you can close everything, you are now ready to use the program !

### Installation troubleshooting

The project works on Linux, macOS, and Windows.
On Windows, installation can be a bit more difficult because of Conda and terminal settings.
This section helps you fix the most common problems.

#### Troubleshooting step 1 : Make sure Conda is available (Windows users)

On Windows, the `conda` command may not work in all terminals.
This often happens if Conda was installed without adding it to the PATH.

If `conda` is not recognized:

Option A (recommended):
Open **Miniconda Prompt** from the Start menu.

Miniconda Prompt is a terminal that is already connected to Conda.
In this terminal, the `conda` command always works and no extra setup is needed.

When it opens, you should see `(base)` at the beginning of the line.
This means Conda is ready to use.

Option B:
Activate Conda manually by running in a **Windows Powershell**:

```bash
C:\Users\YOUR_USERNAME\miniconda3\Scripts\activate
```

If this works, you should see `(base)` at the beginning of the terminal line.

If you prefer to use PowerShell, cmd, or the VS Code terminal,
you may need to run `conda init` once and restart your terminal.
This step is not required when using Miniconda Prompt.

Why this step is useful:
Conda environments only work after Conda itself is active.

#### Troubleshooting step 2 : Check that the environment exists

Open a terminal where Conda works and run:

```bash
conda info --envs
```

This shows all Conda environments on your computer.

- If you see `colony`, the environment was created correctly.
- If you do NOT see `colony`, the environment was not created.
  In that case, go back to the installation steps and create it again.

Why this step is useful:
It checks that the project environment really exists before trying to use it.

#### Troubleshooting step 3 : Activate the project environment

Once Conda is active, run:

conda activate colony

If it works, your terminal line will start with `(colony)`.

If you do not see `(colony)`, the environment is not active and the program will not work.

Why this step is useful:
The `colony` environment contains the correct Python version and libraries for this project.

#### Troubleshooting step 4 : PowerShell security message (Windows only)

Some Windows computers block scripts by default.
If you see a message saying that scripts are disabled, run this command once in a **Windows Powershell**:

Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

Then close the terminal and open it again.

Why this step is useful:
It allows Conda to run safely in the terminal.

#### Troubleshooting step 5 : Try again

After fixing the problem, run again:

```bash
conda activate colony
```

If `(colony)` appears, everything is ready.

#### Quick checklist before running the program

- `conda info --envs` shows `colony`
- The terminal line starts with `(colony)` after the command `conda activate colony`
- Running `python --version` works

If all three are true, the installation is finished and you can run the program.


---

## 1. Usage tutorial

All you will need to touch will be inside the `usage` folder. 

### 1) Put your images in the `usage/images` folder (create it if needed)

#### Quick checklist before putting the images

- verify that the images names start with "plate_x" (x being a number)
- verify that the images are plates with 12 wells and are cropped correctly
- ensure that the image quality is not poor or blurry
- ensure the format of the image is
  - .jpg
  - .jpeg
  - .png

### 2) Run the pipeline

Open a terminal in the `usage` folder. To do so, you can right-click on the `usage` folder and select “New Terminal”.

Before the first counting of each session, you will need to **activate** the Python environment by running this in the terminal :

```bash
conda activate colony
```

After activating the environment, your terminal prompt should start with (colony), for example:
```
(colony) user@computer:~/.../usage$
```

**Now you can count your colonies !** 

Run the script by using this command :

```bash
python count_colonies.py
```

If some images are too complex to crop automatically, the program will tell you which ones.
These images must be cropped manually (see cropped image example bellow), then placed back into usage/images before running the script again.

### 3) Check your results !

After the script finishes successfully, the results are saved in the `usage` folder: `usage/results.xlsx`.

Open this Excel file to see the output.

The file contains, for each image :
- the image name
- a table with identified wells (A1, A2,...)
- the predicted number of colonies or uncountable(if the well is too full or a quality/image issue)

An example with this image (cropped version) : 

![Example plate](data/cropped_data/PR8/Plate_19.jpg)

The output will look like this in a table in the Excel file :

**example_plate.jpg** (Image file name) 
| 4 | 3 | 2 | 1 |  |
|---|---|---|---|---|
| 0 | 4 | 44 | uncountable | **A** |
| 0 | 7 | 50 | uncountable | **B** |
| 1 | 8 | 49 | uncountable | **C** |

#### Problem during usage 
If `conda activate colony` does not work, try to check that the environment exists:
```bash
conda info --envs
```
You should see `colony` in the list of environments.

- If `colony` is listed, try activating it again:
  ```bash
  conda activate colony
  ```

- If `colony` is **not** listed, the environment was not created correctly (see "Installation and Setup" section)

## 2. Advanced usage: ML personalization

This section is intended for advanced users who want to modify, retrain, or extend the machine-learning models.  
It is **not required** to use the colony counting tool.

### Adding a new model

The project uses a central `MODEL_DICTIONARY` to register all available neural network architectures.  
Each entry defines:

- the **model class**  
- the **constructor arguments** (`kwargs`)  
- the **default weight file** to load/save  

This makes it possible to select any model using a simple CLI flag like:
```bash
--model EfficientNet
```

To add a new model, create your model class in `src/ml/models`. All models must inherit from `torch.nn.Module`, which is the standard base class for neural networks in PyTorch.


This ensures that the model:
- is recognized as a trainable network,
- registers its layers and parameters,
- can be trained, evaluated, saved, and loaded easily by the pipeline.

A valid model must define an `__init__` method and a `forward()` method that specifies how input images are processed.

Once implemented, insert then a new entry in the dictionary for your model:

```
"my_new_model": {
    "class": MyModelClass,
    "kwargs": {"any_construction_param": xyz},
    "weights": "my_new_model.pth"
}
```

### Customizing data transformations

It is also possible to modify the **image transformations** applied to the data before training or inference.

These transformations simulate variations in images, such as:
- slight blur (e.g. Gaussian blur),
- changes in brightness or contrast,...

By adjusting these transformations, it is possible to:
- make the model more robust to noisy or imperfect images,
- adapt the model to specific acquisition setups,
- experiment with how different image qualities affect performance.

All transformations are defined in the `src/ml/data/transforms.py` file and are applied automatically to the data in the code.

The project uses **different image transformations depending on the context** in which the images are used.

- **Training transformations** are used when training the models.
  They include small random variations (such as rotations or blur) to expose the model to a wider range of image conditions and improve robustness.

- **Test (evaluation) transformations** are more conservative.
  They do not introduce randomness and aim to reflect real input images as closely as possible.

Different transformations are also applied depending on the model:

- The **classifier transforms** help the classifier to detect if a well is too dense or not to count it.
  Its transformations focuses on global structure and overall density.

- The **counter transforms** help the counter to detect the colonies.
  Its transformations focuses on local contrast, colors difference,...

As a result, the classifier and the counter each use **their own transformation pipelines**, adapted to their specific tasks.

You can check how your transforms look on an image by running the `plt_transform.py` file.  
Add to `IMG_PATHS` the paths to the well images you want to see the effect of the transformations on.  
Import the transforms you want to test in the file and apply them.

It will output 3 transformed wells so, if you have random transforms, you can see different variations.

### Adding new data to the project

This section explains how to add **new plate images** so they can be used for training or evaluation.

---

#### 1. Prepare the raw data

To add new data, you need:
- raw images of **12-well plates**,  
- an **Excel file** containing the plate counts for each well.

The Excel file must follow the same structure and naming conventions as the existing ones.

Place:
- the raw plate images in a **new subfolder** inside `data/raw/` and named after your plate type,
- the corresponding Excel file in the same location.

Keeping consistent filenames and format like the other data is important so images and labels can be matched automatically.

#### 2. Run the preprocessing pipeline

Once the new data is added, run the preprocessing scripts in the following order:

1. **Crop plates**  
   Extract the 12-well plate from each raw image:
   ```bash
   python src/preprocessing/crop_plates.py
   ```
2. **Crop wells**  
   Split each plate into individual well images:
   ```bash
   python src/preprocessing/crop_wells.py
   ```
3. **Build the dataset**  
   Link cropped wells with their plaque counts and generate the dataset CSV files:
   ```bash
   python src/preprocessing/build_dataset.py
   ```

4. (Optional) Apply data augmentation
  If you want to increase the dataset size and reduce class imbalance, you can run the data augmentation step:
   ```bash
   python src/preprocessing/data_augmentation.py
   ```
  This will generate additional transformed versions of the wells (rotations, flips, etc.).

After augmentation, run `build_dataset.py` again to include the augmented samples in the augmented dataset.

### Training

Models are trained **only on the training set**, never seeing the test samples.

During training:

- **cross-validation** is used to evaluate model stability,  
- the model is trained for several epochs,  
- **the best epoch (lowest validation loss)** is saved,  
- this checkpoint is later used for evaluation.

This ensures that the final evaluation truly reflects performance on **unseen data**, avoiding overfitting.

#### 3.1 Train the Countability Classifier

The classifier determines whether a well is **countable or uncountable** (the number of epochs here is just an example).

```bash
python src/ml/training_classifier.py \
    --csv data/training.csv \
    --epochs 20
```

#### 3.2 Train the Colony Counter (Regression)

The counter model predicts the **number of colonies** inside countable wells (the number of epochs here is just an example).

```bash
python src/ml/training_counter.py \
    --model ResNet34 \
    --epochs 30 
```

You may choose any model registered in `MODEL_DICTIONARY`:

```bash
--model EfficientNet
--model ResNet34
--model ColonyCNN
```

### Evaluation

To evaluate the model performance, the dataset is split into **training** and **test** sets while preserving the natural distribution of:

- **countable vs uncountable** wells  
- **zero-colony vs non-zero** wells  

This stratified split ensures that the test set is representative.

#### 4.1 Running the Full Evaluation Pipeline

Once the classifier and counter models are trained, you can run the evaluation of the pipeline (you can use the test_augmented.csv too):

```bash
python src/ml/evaluate_pipeline.py \
    --csv data/test.csv \
    --counter_model ResNet34
```
The evaluation script then performs the classification then  counting and outputs metrics to give an insight of performances and saves the predictions in a CSV file.

#### 4.2 Adapting the `count_colonies.py` script

Load your own model in the script by changing this part of the code, replace ResNet by yours :

```bash
counter = ResNet34Regressor(pretrained=True, dropout_p=0.5, freeze_backbone=False)
counter.load_state_dict(torch.load(COUNTER_WEIGHTS_PATH, map_location=device, weights_only=True))
counter.to(device)
counter.eval()
```

## 3. Data Overview

The dataset consists of images of plates, each containing **12 individual wells**.  
This is an example of a raw image received from the lab:

![Raw plate](data/raw/IBV/Plate_1.jpg)

To prepare the data for training or inference, two preprocessing steps are applied:

### **1. Plate Cropping**
Full images often include some background.
A first cropping step takes out all the background and keep just the plate.

![Cropped plate](data/cropped_data/IBV/Plate_1.jpg)

### **2. Well Cropping**
Each plate contains 12 wells arranged in a 3x4 grid.  
For every plate image:

- the position of each well is kept(e.g A1,B1,A2,...),
- each well is cropped into its own image,
- resulting in a dataset where **each row corresponds to a single well**.

For example here the A1 well cropped (top-right well) :
![Cropped well](data/cropped_wells/IBV/plate_1/IBV_plate1_A1.jpg)

### **Labels**
Each well has:

- `is_countable` : 1 if the well is countable, 0 otherwise,  
- `value` : number of colonies (or -1 for uncountable wells).

### Data Augmentation

Because the original dataset was relatively small, data augmentation was used to improve model robustness and prevent overfitting.  
The augmented dataset is stored in a separate folder (`data/augmented_wells`) and is generated from the original well images.

The main goals of augmentation were:

- increase the dataset size,  
- expose the model to realistic variations in orientation,  
- handle class imbalance of wells with colony count of **0**.

### Data Analysis Notebook

To better understand the dataset, a Jupyter notebook (`data_analysis.ipynb`) is provided.  
It performs an exploratory analysis of:

#### The Raw Dataset
- distribution of colony counts in general and amongst types of plates
- proportion of **countable vs uncountable** wells  
This helps identify biases or irregularities before training.

#### The Augmented Dataset
Because the dataset is small, an augmented dataset is generated.  
The notebook allows you to:

- inspect how many augmented samples were generated per class  
- see the new augmented distributions  
- verify that **stratified augmentation** reduces label imbalance  

This ensures the augmentation strategy is improving data diversity and quantity.

#### Model Predictions (Post-Evaluation)
After running the full pipeline, the notebook can also:

- load `pipeline_results.csv`  
- show the distribution of predictions
- plot predictions that stays within the range of 10% of allowed error  
- compute prediction patterns per plate type 

This allows a better understanding of **where the model performs well** and **where improvements may be needed**.

---

Link to old repo for the project :
https://github.com/JulienErblandEPFL/Colony_counter

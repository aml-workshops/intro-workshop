# Creating Azure Machine Learning Compute Instance

Compute instances (CI) are user-specific, Linux-based virtual machines (VM) that allow you to take advantage of compute in the cloud. You can interact with these instances via Jupyter, RStudio, VSCode, or other SSH tools.

For this lab, we'll be leveraging CIs to streamline package management and provide a consistent environment to learn in. In this lab, each user will be creating their own CI.

The steps to create the CI are:
1. Navigate to https://ml.azure.com and login with your Active Directory (AD) credentials.
1. Click on 'Compute' in the left navigation pane.
    ![Compute](../../media/1-compute-navigation.jpg)
1. In the 'Compute Instances' tab (1), click on '+ New' (2)
    ![New Compute Instance](../../media/2-ci-new.jpg)
# Linear Regression for price prediction
This portfolio project demonstrates my basic data sciece skills.
It follows the dataset analysis from Chapter 1 form "Hands-On Machine Learning" by A. Geron (which I'm carefully studying for the second time)

Instead of picking some dataset from Kaggle and doing some random manipulatiuons to sqeeze out some insight from it, I decided to solidify my skills by following the the workflow or a professional data scientist.



## Jupyter notebook with EDA:
[exploratory_data_analysis.ipynb](/exploratory_data_analysis.ipynb)



## The usual "shell preparations":
```shell
# getting started
$ git init myrepo
$ cd myrepo
$ touch README.md requirements.txt
$ echo "# todo" > README.md
$ vim requirements.txt
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ git add README.md requirements.txt
$ git commit -m “my first commit”

# create a new repo on GitHub; generate a token
$ git remote add origin https://github.com/leztien/myrepo.git
$ git branch -M main
$ git pull origin main --rebase  #??
$ git push --set-upstream origin main

# routine commits
$ git add README.md requirements.txt
$ git commit -m "comment"
$ git push

# deployment
$ pip freeze > requirements.txt
$ copy the contents of this folder into a new 'deployment' folder and cd into it
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ uvicorn main:app --reload
```



<br><br><br>
some links:
##### Hands-On ML github:
https://github.com/leztien/handson-ml3-forked

##### A.Geron's notebook on Chapter 2:
https://colab.research.google.com/github/ageron/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb

##### link to my github repo of this project:
https://github.com/leztien/portfolio-project-linear-regression-for-price-prediction/tree/main

...

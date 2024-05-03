# Protocol to perform a metric evaluation

The evaluation of the metric takes a lot of time so it's important to do it with great care to evade to lose a lot of time.
This file contain a precise protocol to follow that should hopefully make this process go right.

1. Make sure that all documents have been indexed\
    You can check this by searching for `www angel families city slide com` in the search engine and it should return the document `d-423636` quite high in the (unranked) results.
2. Reset your dataset\
    To make sure that everyone uses the same dataset, remove your existing one and unzip the original one again:
```
rm datasets/AOL4PS/training_data.csv
rm datasets/AOL4PS/validation_data.csv
```
3. Select the metric you will be evaluating\
    In `PIR.py` go line 34 and uncomment the line with the metric you will evaluate. Make sure to have only the good metrics selected, having too much can extends significantly the evaluation time.
4. Perform validation\
    Before starting the validation, send a message to say that you are doing it so that no clash appear.
```
./PIR.py -v
```
5. Save and share the result\
    Once the evaluation is finished, a file called `evaluation_results.json` should appear at the root of the project.
    Make sure that this file contain evaluation metrics for both "logs" and "elastic search".
    Then rename it with the name of the evaluated metrics, commit the file and push it.

Thank you =)
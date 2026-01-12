# F2-Score Verification Report

This report summarizes the F2-Score results for the different test cases.

## Case 1_1: Single Prediction, Single Ground Truth

| Subcase     | TP | FP | FN | Precision               | Recall                  | F2-Score                | GT Coordinates | PR Coordinates | Description                               |
| :---------- | :- | :- | :- | :---------------------- | :---------------------- | :---------------------- | :--- | :--- | :---------------------------------------- |
| little_match| 1  | 0  | 0  | 0.14285714285387754     | 0.14285714285387754     | 0.14285714285387754     | (100,100,50,50) | (125,125,50,50) | Prediction and ground truth slightly overlap. |
| big_match   | 1  | 0  | 0  | 0.9238168526000841      | 0.9238168526000841      | 0.9238168526000841      | (100,100,50,50) | (101,101,50,50) | Prediction and ground truth largely overlap.  |
| max_match   | 1  | 0  | 0  | 0.9999999999600001      | 0.9999999999600001      | 0.99999999996           | (100,100,50,50) | (100,100,50,50) | Prediction and ground truth perfectly overlap.|
| none_match  | 0  | 1  | 1  | 0.0                     | 0.0                     | 0                       | (100,100,50,50) | (150,150,50,50) | Prediction and ground truth do not overlap.   |

## Case 1_2: False Positive (FP)

| TP | FP | FN | Precision           | Recall             | F2-Score           | GT Coordinates | PR Coordinates | Description                               |
| :- | :- | :- | :------------------ | :----------------- | :----------------- | :--- | :--- | :---------------------------------------- |
| 1  | 1  | 0  | 0.46190842630004203 | 0.9238168526000841 | 0.7698473771667368 | (100,100,50,50) | (101,101,50,50), (300,300,10,10) | A prediction exists where there is no ground truth. |

## Case 2_1: False Negative (FN)

| TP | FP | FN | Precision           | Recall              | F2-Score           | GT Coordinates | PR Coordinates | Description                               |
| :- | :- | :- | :------------------ | :------------------ | :----------------- | :--- | :--- | :---------------------------------------- |
| 1  | 0  | 1  | 0.9238168526000841  | 0.46190842630004203 | 0.5132315847778245 | (100,100,50,50), (200,200,20,20) | (101,101,50,50) | A ground truth exists where there is no prediction. |

## Case joongbok: Complex Case (Duplicate FP + FN)

| TP | FP | FN | Precision           | Recall              | F2-Score            | GT Coordinates | PR Coordinates | Description                               |
| :- | :- | :- | :------------------ | :------------------ | :------------------ | :--- | :--- | :---------------------------------------- |
| 1  | 1  | 1  | 0.3403361344423417  | 0.3403361344423417  | 0.34033613444234173 | (100,100,50,50), (200,200,50,50) | (105,105,50,50), (95,95,50,50) | One ground truth is predicted multiple times (duplicate FP), and another ground truth is missed (FN). |

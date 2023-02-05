# Classification and Analysis of Yoga asana
+ The project idea is to develop a system which correctly classifies the following six yoga asana : Bhujangasana, Padmasana, Shavasana, Tadasana, Trikonasana and Vrikshasana and suggest scope of improvement for the same. 
+ Created the dataset from scratch by recording videos of the six yoga asanas from various yoga classes within the city.
+ The user inputs a video.Frames are extracted from the video uploaded. 'Open Pose' which is a pre-trained CNN, is used to detect the keypoints of the joints detected in each frame of the video. These keypoints are passed on to a LSTM model for recognizing the time sequence and predicting the asana.The angle is also calculated between the joints and compared with that of an expert's for analysis.
+ The analysis of the asana is shown to the user through an image which is the last frame from the video.This image shows the angles specific to the predicted asana between relevant joints of the body. If the asana is found to be correctly done, then the angles are shown green in colour, else, they are shown in red.  


**Front-end of the application**  

 ![HomrePage](https://user-images.githubusercontent.com/54678638/216806390-1a198f09-a87c-48b6-93ec-da8bb7e0fc06.png)  
 
 ![UploadVideo](https://user-images.githubusercontent.com/54678638/216806411-3a8f7d1b-68b9-43eb-b9e1-d623e5385a60.png)  

**Output**  

![Analysis-Corrrect](https://user-images.githubusercontent.com/54678638/216806438-c2241b34-1977-49e8-bf96-33a2ee4cc72c.png)  

![Abalysis-Incorrect](https://user-images.githubusercontent.com/54678638/216806461-e19d6f5c-4732-4622-9717-271701c71af5.png)





**Player Re-Identification Report (Human Writeup)**

so for this project i was working on task 2 from the Liat.ai intern assignment, which was to re-identify players in a 15-second soccer video. the idea was to give consistent IDs to players (like Player\_1, Player\_2 etc) even if they go off screen and come back again. i used YOLOv11 for detection and DeepSORT for tracking. all of this was done on my GPU setup (RTX 3050 6GB, CUDA 12.8), ran pretty okay on windows 11.

for detection, i used the pretrained model (best.pt) and ran it on each frame with confidence 0.2 and iou 0.5. initially had conf=0.5 but it missed lot of players especially those far or bit occluded. reducing it helped bring up detection count to like 8-12 players per frame. still some were missed tho. i also added player\_class\_id filtering based on the model class but still need to confirm if class 0 is actually the player in the model, didn't check model.names properly yet

for tracking, DeepSORT was used with MobileNet embeddings. started with max\_age=30, n\_init=3 but tracks were getting dropped quickly so i changed to max\_age=70 and n\_init=1, also lowered iou\_dist to 0.5. helped in keeping more consistent IDs but still some tracks stayed tentative. added some debug prints to see detection and tracking counts and confirmed the main issue was not enough detections happening for every player

the performance was pretty decent, around 25 to 30 FPS on GPU. some people suggested resizing to 640x360 for speed but i didn’t do it yet. maybe later. bounding boxes were drawn with green for confirmed and red for tentative tracks. output video and tracking\_data.csv were generated as expected.


**For model development i used the "SIMPLE ONLINE AND REAL TIME TRACKING WITH A DEEP ASSOCIATION METRIC " Research paper as reference** 

This paper leverages DEEP-SORT mechanism in order to track objects. This paper is brief and helped me alot in order  to complete this project.
REFERENCE PAPER : [https://arxiv.org/abs/1703.07402] 

**challenges**

so yeah some players are still missed especially those in the background or near goal post where it gets crowded. also model might not be trained on videos like this one so it probably struggles. tracking also isn’t perfect coz when detection fails DeepSORT drops the player even if it’s same one. i think better embeddings or more fine-tuned model would help here. class ID also still a bit confusing, i should double check what class best.pt actually detects.

**incomplete stuff and next plans**

* not all players are detected and tagged with IDs
* need to confirm correct player\_class\_id using model.names
* maybe reduce conf even more to like 0.1 but might get false positives
* could try ResNet for better embeddings in DeepSORT
* sharpen or apply super res on video maybe? might help detect small players

if had more time/resources, would def fine-tune YOLOv11 on soccer video dataset with similar conditions as this one. also try other trackers like ByteTrack or OC-SORT, i heard they work better with occlusions. and maybe add optical flow for smoother tracking. and yeah if i had like RTX 3080 or something this whole thing would just fly lol


**The ouput i got after testing mmodel**
Processing complete. Total time: 82.94 seconds
Output saved: output.mp4, tracking_data.csv
Total frames processed: 375
Total detections: 6579
Average detections per frame: 17.54
Average detection confidence: 204.026



**Conclusion**

overall the pipeline is like halfway there. it detects and tracks most players okay, but still not 100% reliable yet. just need to tune it a bit more. documentation is done, README’s ready and the output video is generated. just need to fix that last bit of debugging and then it's all good to go.




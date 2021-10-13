# MaskDetection
a software that recognizes people that does not put their mask on properly and alerts

# Final Project Software Engineering Idan and Ofek 2021



## overall 

### description about the project : 

- who is the client? : the client is organization that wants to use the product : the project will be a  software that the client will be able to download , and use . 

- what does the project do : we want to be able to **detect** people that remove/ don't put on properly their mask in a public setting like theater / class  . the software will only be able to **detect** whether or not a person has it's mask , **and** **wont** **be able to detect faces for various reasons that make this ability unfeasible ** . if the public setting is known , then the software will be able to know where exactly that person is sitting , and will be able to classify him ( the software will be able to detect the seat , and in a case like theater , the people's seat is determined ahead of time/ in case like a classroom , the software will be able to show him , and know where he is seating  ).

- in order to know where the camera is placed (in each public setting ) , the software will have an initialization setting where it will be able to detect all of the seats and rows . 

- the software will be located on a pc that is connected to a camera (that has the requirements specified below)

- the software will have an UI that will give easy and quick control over the project's features 

  

  

  ## features : 

  - detect who has and how doesn't have a mask on properly
  - count how much time the person had his mask in a certain period of time 
  - detect where he is sitting , and be able to show his face 
  - detect how many people are in the room 
  - the software will have a FPR , TNR that will be able to be changed by the user . (what is the accuracy , precision - control the threshold  )





## requirements : 


![image-20210928214017604](https://user-images.githubusercontent.com/80175752/137226152-cbe71985-2214-4012-920e-a213aa8c08ac.png)



- **detecting** the presence of a person in a scene might require that the height of the person occupies 20% of the view. **Recognizing** a person, however, might require that the person occupies 40%, and **identification** person might require 140% or more (i.e. the person is taller than the image)."



![image-20210928214536037](https://user-images.githubusercontent.com/80175752/137225901-e6c84ed7-97b5-434c-b76c-b84709569bee.png)


- **Camera** positioning is critical for successful identification. This is not only to avoid difficult lighting situations, but also to ensure that persons or objects are captured at a favorable angle. A birds-eye perspective from a camera placed high above the ground will cause some degree of distortion, making it difficult to identify persons or objects.










![image-20210928215828526](https://user-images.githubusercontent.com/80175752/137225906-ed134809-b5f4-42fc-a91c-0334cf5bf0ad.png)


![image-20210928222045600](https://user-images.githubusercontent.com/80175752/137225908-d15320ff-df7d-4883-8a11-da5c31bd1a55.png)

## specification requirements 

- [CCTV - closed-circuit television options and resolutions  ]("https://www.axis.com/products/product-selector#!/")
- specified as above , sufficient focal length , resolution , LUX amount , angle , blurriness  , FPS .





## important details 

### Crucial Problems : 

- Camera placement or lens selection that distorts facial features (the angle of the camera if its placed at the celling )
- Difficult lighting conditions that create shaded areas or whiteout effects (like in cinemas and theaters)
- Compression settings that cause image blur or pixelation (if the camera doesnt have the sufficient resolution required / focal length )
- Motion blur caused by slow shutter speeds or low frame rates (FPS of the camera )
- Excessive noise in low-light situations (theater , classroom with not enough light )
- Overlay text appearing in a crucial part of the scene (someone is walking by )



### question for Rachel : 

- where can we get data to test on 

- where and when do we get our camera 

- do we have a budget 

- can we even place a surveillance camera in a public setting like a classroom 

- who can we go to with our questions ? 

- can we design our UI / UX or do we have to follow a specific guideline 

  


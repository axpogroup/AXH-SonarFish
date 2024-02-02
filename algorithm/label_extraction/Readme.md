_Code written by Leiv Andresen, HTD-A, leiv.andresen@axpo.com
January 2023_


TODO BASTIAN: ADD something here



_This code is not written or documented for reuse out of the box. Certain settings might will need to be adjusted for successful operation._

The goal is to extract the position of fishes and their interactions with the Rake (Rechen) from ARIS sonar imagery.
To this end HTU marked the relevant content with a program: boxes of different colors (red, blue, green) symbolize fish, green and red dots mark interactions with the rake.
The following code is based on the sonar fish detection algorithm and detects the location of the boxes and dots. 


# Optimal Acoustic Coding

The goal of this project is to examine communication in the presence of natural environment noise. Initially investigating human speech (english) with varying levels of background noise inspired by the street, the plan is to extend this to animal sounds in different noisy channels (whales in the ocean vs birds in the forest), as well as investigate synthetic sounds. 

The idea is that in some way, communication was developed in response to be robust to the environment in which it is used. This means that some methods may be more or less robust to qualities to reverberance and non-natural noises.

In order to test this, we take a clean noiseless audio you may hear from a studio environment as our message, and then we add in noise to create a layered sound. Then we'll use computational methods to see if parts or the entire message can be decoded despite the added noise. In our initial example, a recording of someone counting from 0 to 12, this computational method can try to identify phonemes, words, or other parts. From this we can calculate Bit Error Rate Graphs, and have an idea of the potential of each method. Ideally, whenever we either use non-natural noise, or edit the communication, it should perform as good or worse than the original method.

import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import matplotlib.pyplot as plt

from game.flappy_bird import GameState


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        # channels/features, output channels/feature, kernels, stride

        # 1st CNN
        # self.conv1 = nn.Conv2d(4, 32, 8, 4)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(32, 64, 4, 2)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv3 = nn.Conv2d(64, 64, 3, 1) #4*3*3*64
        # self.relu3 = nn.ReLU(inplace=True)
        # self.fc4 = nn.Linear(3136, 512) # how many outputs come out = 3136
        # self.relu4 = nn.ReLU(inplace=True)
        # self.fc5 = nn.Linear(512, self.number_of_actions)

        # 2nd CNN
        # self.flatten = nn.Flatten()
        # self.conv1 = nn.Conv2d(4, 16, 9, 5, bias=False, padding="valid")
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(16, 32, 5, 3, bias=False, padding="valid")
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv3 = nn.Conv2d(32, 32, 3, 1, bias=False, padding="valid")
        # self.relu3 = nn.ReLU(inplace=True)
        # self.fc4 = nn.Linear(128, 256)
        # self.relu4 = nn.ReLU(inplace=True)
        # self.fc5 = nn.Linear(256, self.number_of_actions)

        # 3rd CNN
        # self.flatten = nn.Flatten()
        # self.conv1 = nn.Conv2d(4, 16, 9, 5, bias=False, padding="valid")
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(16, 32, 3, 1, bias=False, padding="valid")
        # self.relu2 = nn.ReLU(inplace=True)
        # self.fc4 = nn.Linear(6272, 64)
        # self.relu4 = nn.ReLU(inplace=True)
        # self.fc5 = nn.Linear(64, self.number_of_actions)

        # 4th CNN
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(4, 16, 9, 7, bias=False, padding="valid")
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, bias=False, padding="valid")
        self.relu2 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(2592, 32)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(32, self.number_of_actions)



    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        # out = self.conv3(out) #Comment out for third and fourth CNN's
        # out = self.relu3(out) #Comment out for third and fourth CNN's
        # out = out.view(out.size()[0], -1) #Comment out for second, third and fourth CNN's
        out = self.flatten(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        # m.bias.data.fill_(0.01)
    if type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        print("CUDA available")
        image_tensor = image_tensor.cuda()
    return image_tensor


def resize_and_bgr2gray(image):
    image = image[0:288, 0:404] #change first 0 to 50 to crop behind the bird
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1)) #image size
    return image_data


def train(model, start):
    # define Adam optimizer
    summary(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    episode = 0
    max = 0
    maxReward = 0
    episodeReward = 0
    countList = []
    rewardList = []

    # initialize mean squared error loss
    criterion = nn.MSELoss()

    # instantiate game
    game_state = GameState()

    # initialize replay memory
    replay_memory = []

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal, score = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    # initialize epsilon value
    epsilon = model.initial_epsilon
    iteration = 0

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    # main infinite loop
    while iteration < model.number_of_iterations:
        # get output from the neural network
        output = model(state)[0]

        # initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        # if random_action:
            # print("Performed random action!")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()

        action[action_index] = 1

        # get next state and reward
        image_data_1, reward, terminal, score = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        #add the reward for this state to the end
        episodeReward += reward

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # save transition to replay memory
        replay_memory.append((state, action, reward, state_1, terminal))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        # unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # get output for the next state
        output_1_batch = model(state_1_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        # extract Q-value
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = criterion(q_value, y_batch)

        # do backward pass
        loss.backward()
        optimizer.step()

        # set state to be state_1
        state = state_1
        iteration += 1


        if iteration % 25000 == 0:
            print("SAVING MODEL")
            torch.save(model, "pretrained_model/current_model_" + str(iteration) + ".pth")
            plt.figure(figsize=(10, 10))

        if terminal:
            episode += 1
            rewardList.append(episodeReward)
            countList.append(episode)
            if score > max:
                max = score
            if episodeReward > maxReward:
                maxReward = episodeReward

            print("----------------------------")
            print(f"EPISODE: {episode}")
            print(f"FINAL SCORE: {score}")
            print(f"MAX SCORE: {max}")
            print(f"EPISODE REWARD: {episodeReward}")
            print(f"MAX REWARD: {maxReward}")
            print("iteration: ", iteration, "\nelapsed time:", time.time() - start, "\nepsilon:", epsilon,
                  "\nQ max:", np.max(output.cpu().detach().numpy()))
            print("----------------------------")

            episodeReward = 0
            if episode % 10 == 0:
                plt.clf()
                plt.plot(countList, rewardList)
                plt.xlabel("Episodes")
                plt.ylabel("Reward")
                plt.suptitle("Rewards: Normal Level")
                plt.title("4th CNN")
                plt.savefig("images/rewards.png")
                plt.clf()


def test(model):
    game_state = GameState()

    histList = [0] * 20
    scoreList = [0] * 20
    histRange = range(0,20)
    episodeReward = 0
    episode = 0

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal, score = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    episodeReward += reward

    while True:
        # get output from the neural network
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()
        action[action_index] = 1

        # get next state
        image_data_1, reward, terminal, score = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1

        #add iteration reward to episode reward
        episodeReward += reward


        if terminal:
            if reward > 90:
                score = 20
            print(score)
            scoreList[score] += 1

            #Reset episodeReward now that it's over
            episodeReward = 0
            episode += 1

            if episode % 10 == 0:
                plt.clf()
                plt.bar(histRange, scoreList)
                plt.xlabel("Final Score")
                plt.ylabel("Episode Count")
                plt.suptitle("Scores: Normal to Difficult")
                plt.title("4th CNN")
                plt.savefig("images/Score Histogram.png")
                plt.clf()

            if episode == 1000:
                print("Reached 1000 episodes")
                break



def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load(
            'pretrained_model/4th CNN (normal).pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        test(model) #comment to continue training off of a previous model
        # uncomment to continue training off of a previous model
        # start = time.time()
        # train(model, start)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        model = NeuralNetwork()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()

        train(model, start)


def imshow(img, title):
    """Custom function to display the image using matplotlib"""

    # define std correction to be made
    std_correction = np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    # define mean correction to be made
    mean_correction = np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1)

    # convert the tensor img to numpy img and de normalize
    npimg = np.multiply(img.numpy(), std_correction) + mean_correction

    # plot the numpy image
    # plt.figure(figsize=(64 * 4, 4))
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.savefig("images/CNN Current State image")

def saveCNNimg(model):
    model_weights = []
    conv_layers = []
    model_children = list(model.children())
    # print(model_children)
    counter = 0
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        print(type(model_children[i]))
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
            break
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolutional layers: {counter}")
    plt.figure(figsize=(20, 17))
    for i, filter in enumerate(model_weights[0]):
        plt.subplot(8, 8,
                    i + 1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].detach(), cmap='gray')
        plt.axis('off')
    plt.savefig("images/CNN image")
    exit()


if __name__ == "__main__":
    main("train")

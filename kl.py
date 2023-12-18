# import torch
# from torch.nn import CrossEntropyLoss
# # def kl_divergence(q_batch_mu, q_batch_var, p_batch_mu, p_batch_var):
# #     # between N(mu_1,sigma^2_1) and N(mu_2,sigma^2_2)
# #     # D(q||p) = q log(q/p)
# #     term1 = torch.sum(0.5 * (p_batch_var - q_batch_var))
# #     term2 = torch.sum((torch.exp(q_batch_var)+torch.pow(q_batch_mu-p_batch_mu,2))/(2*torch.exp(p_batch_var))-0.5)
# #     return (term1+term2)/q_batch_var.size(0)
# #
# #
# # def KLD(z_mu, z_logvar):
# #     # between N(0,I) gaussian and z gaussian
# #     kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
# #     return kld_loss / z_mu.size(0)
# #
# # q_mu = torch.tensor([0.0])
# # q_var = torch.tensor([4.0])
# #
# # p_mu = torch.tensor([0.0])
# # p_var = torch.tensor([0.0])
# #
# # #
# # #
# # # print(kl_divergence(q_mu,q_var,p_mu,p_var))
# # # print(KLD(q_mu,q_var))
# # loss_func = CrossEntropyLoss()
# # q = torch.tensor([[0.1,0.1,0.8]])
# # p = torch.tensor([[0.1,0.1,0.8]])
# #
# # print(loss_func(p,q))
#
# import torch
# import numpy as np
# from torch.distributions import Categorical, kl
#
# # KL divergence
# p = [0.1, 0.2, 0.3, 0.4]
# q = [0.1, 0.1, 0.7, 0.1]
#
# def cross_entropy(q_prob,p_prob):
#     a=torch.sum(q_prob * torch.log(q_prob/p_prob))
#     return a
#
# Dpq = sum([p[i] * np.log(p[i] / q[i]) for i in range(len(p))])
# print (f"D(p, q) = {Dpq}")
# dist_p = Categorical(torch.tensor(p))
# dist_q = Categorical(torch.tensor(q))
# print(cross_entropy(torch.tensor(p),torch.tensor(q)))
# print (f"Torch D(p, q) = {kl.kl_divergence(dist_p, dist_q)}")

# value_list = []
# for i in range(0,5):
#     a = [1,2,3]
#     value_list.append(a)

# import torch
#
# # Create a 1D tensor
# a = torch.randn([80,10,10])
#
# b = [a,a,a,a,a]
# c = torch.hstack(b)
# print(c.shape)
import torch

# a = torch.rand((20000, 5))  # create a random tensor
# b = torch.randint(5, size=(20000,))  # create a random tensor with values in [0,4]
# print(a.shape)
# print(b.shape)
# c = torch.gather(a, 1, b.unsqueeze(1)).squeeze()  # select the values in a corresponding to the indices in b
#
db = 20 * torch.log10(torch.tensor(1/0.1))
print(db)
x = [10,12,14,16,18,20]
a = [1/(10**(i/20)) for i in x]
print(x)
print(a)
# c = torch.where(b.any(dim=1), indices, torch.tensor(-1))  # set invalid indices to -1
# print(c)

# import torch
#
# b = torch.rand((10, 5))  # create a random tensor
# indices = torch.argmax(b.ne(0) * 1.0, dim=1)  # find the first non-zero index in each row
# c = torch.where(b.any(dim=1), indices, torch.tensor(-1))  # set invalid indices to -1

# import numpy as np
#
# # Example array of digit labels with shape [20000]
# label_array = np.random.randint(low=0, high=10, size=(20000,))
# print(label_array)
# # List of corresponding text labels
# text_labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck','Imagenet Resized']
#
# # Use take() function to replace digit labels with text labels
# text_labels_array = np.take(text_labels, label_array)
#
# # Print the resulting array
# print(text_labels_array)
#
# import seaborn as sns
# import numpy as np
# from matplotlib import pyplot as plt
#
# # Generate some random data
# fig = plt.figure()
# x = np.random.randn(100)
# y = np.random.randn(100)
# labels = np.random.randint(0, 2, size=100)
#
# # Create a scatter plot using Seaborn
# sns.scatterplot(x=x, y=y, hue=labels, palette='Set2', zorder=2)
#
# # Create a second scatter plot for the second class (zorder=1)
# sns.scatterplot(x=x[labels == 1], y=y[labels == 1], color='k', zorder=1)
# plt.show()
# # Show the plot

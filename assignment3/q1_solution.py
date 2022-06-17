import math
import numpy as np
import torch
from torch.autograd import Variable


def log_likelihood_bernoulli(mu, target):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)    # p(x)
    target = target.view(batch_size, -1)

    # log_likelihood_bernoulli
    log_likelihood_bernoulli = (target * torch.log(mu) + (1-target) * torch.log(1-mu)).sum(axis=1)
    
    return log_likelihood_bernoulli


def log_likelihood_normal(mu, logvar, z):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)
    log_norm_constant = -0.5 * np.log(2 * np.pi)

    # log normal
    # if type(logvar) == 'float':
    #   logvar = z.new(1).fill_(logvar)

    a = (z - mu) ** 2
    log_p = -0.5 * (logvar + a / logvar.exp())
    log_p += log_norm_constant
    
    return log_p.sum(axis=1)



def log_mean_exp(y):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param y: (FloatTensor) - shape: (batch_size x sample_size) - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    # init
    batch_size = y.size(0)
    sample_size = y.size(1)

    # log_mean_exp
    a = torch.max(y, dim=1)[0]
    log_mean_exp = torch.log(
    	torch.mean(
    		torch.exp(
    			y-a.unsqueeze(1)
    			)
    		, dim=1)) + a
    
    return log_mean_exp


def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)

    # kld using library functions
    # std_p = np.exp(logvar_p/2)
    # std_q = np.exp(logvar_q/2)

    # p = torch.distributions.Normal(mu_p, std_p)
    # q = torch.distributions.Normal(mu_q, std_q)
    # kld = torch.distributions.kl_divergence(q, p)
    # kld = kld.sum(axis=1)

    # kld using equation 
    kld = ((logvar_q - logvar_p).exp() + ((mu_q - mu_p)**2)/logvar_p.exp() + (logvar_p - logvar_q) - 1)/2
    kld = kld.sum(axis=1)
    
    return kld


def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """

    q = torch.distributions.normal.Normal(mu_q, torch.sqrt(torch.exp(logvar_q)))
    p = torch.distributions.normal.Normal(mu_p, torch.sqrt(torch.exp(logvar_p)))
    z = q.rsample()

    kld = torch.mean(q.log_prob(z) - p.log_prob(z), dim=(1, 2))

    return kld



# GitHub

## Multiple GitHub accounts

You are on your work computer and you want to contribute to both your work and personal/private GitHub repositories. But how to manage multiple identities?

### SSH keys

I assume you aleady have 2 pairs of SSH keys in you `.ssh` folder, a work pair and a private pair. Each pair is associated to a different email address (work and private email addresses). I'll use the following self-explanatory file names for the rest of this tutorial:

```
~/.ssh/work_rsa
~/.ssh/work_rsa.pub
~/.ssh/private_rsa
~/.ssh/private_rsa.pub
```

If you do not have SSH keys already setup, you can for instance use the following command lines to generate them:

```
$ ssh-keygen -t rsa -b 4096 -C "work_email@work.com"
$ ssh-keygen -t rsa -b 4096 -C "private_email@private.com"
```

Then, add your SSH public keys on the different GitHub repositories you want to work with. Make sure not to use your private key for your work repo, and conversly. If you don't know how to add your SSH key to GitHub, follow [this tutorial](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

Finally, add your private SSH keys to your SSH agent using (the `-K` being for Mac users):

```
$ ssh-add -K ~/.ssh/work_rsa
$ ssh-add -K ~/.ssh/private_rsa
```


### Config file

Create a `config` file in your `.ssh` folder. This file should contain something like:

```
# Work
Host github.com
   HostName github.com
   User git
   IdentityFile /Users/vincent/.ssh/work_rsa
   
# Private
Host github.com-private  
   HostName github.com
   User git
   IdentityFile /Users/vincent/.ssh/private_rsa
```

The first part contains the configuration for your work account and the second one for your private account. The specific SSH key is associated to each configuration.


### Clone repository

When cloning a repository, you need to specify if you want to use your private or your work account. Let's say that you want to clone the [From PyTorch to CoreML](https://github.com/vincentfpgarcia/from-pytorch-to-coreml) repository. To clone this repository, GitHub proposes to use the following address (under the Code/SSH section):

```
git@github.com:vincentfpgarcia/from-pytorch-to-coreml.git
```

If you want to clone this repository using your work account, simply use:

```
git clone git@github.com:vincentfpgarcia/from-pytorch-to-coreml.git
```

But if you want to use your private account instead, use:

```
git clone git@github.com-private:vincentfpgarcia/from-pytorch-to-coreml.git
```

Here, I simply replaced `github.com` by `github.com-private` as defined in the `config` file.


### Setup name and email

Finally, if you want to commit in your repository, you need to define the local user's name and email address using (e.g. for a work repository):

```
git config --local user.name "Vincent Garcia"
git config --local user.email "work_email@work.com"
```

And that's it!
# coding: utf-8

# In[1]:


from basic_function import get_train_log

# In[2]:


logs = get_train_log(30000000)

# In[6]:


logs = logs.loc[logs['action_type'] == 0]

# In[13]:


logs = logs.loc[:, ['brand_id', 'seller_id']]

# In[14]:


# In[36]:


import numpy as np

# In[41]:


weight = logs.groupby(by=["brand_id", "seller_id"], as_index=False).agg(np.size)

# In[50]:


weight = weight.reset_index()

# In[51]:


# In[52]:


with open("./file_bs.data", "w") as f:
    for i in weight.itertuples():
        f.write('u' + str(i[1]) + '\ti' + str(i[2]) + '\t' + str(i[3]) + "\n")

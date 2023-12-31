## era_v1_session16
<h2> Description:
 Transformer based encoders are created and used to train and predict.
</h2>
<h4>
Problem statement:<br/>
- Pick the "en-fr" dataset from opus_books<br/>
- Remove all English sentences with more than 150 "tokens"<br/>
- Remove all french sentences where len(fench_sentences) > len(english_sentrnce) + 10<br/>
- Train your own transformer (E-D) (do anything you want, use PyTorch, OCP, PS, AMP, etc), but get your loss under 1.8<br/>
</h4>

<h4> Folder structure</h4> <br/>
|_ runs <br/>
&nbsp;&nbsp;|__ tmodel_150_tokens_bs_48_dff_1024 <br/>
&nbsp;&nbsp;&nbsp;&nbsp;|___ README.md <br/> 
|_ transformer <br/>
&nbsp;&nbsp;|__ config.py <br/>
&nbsp;&nbsp;|__ dataset.py <br/>
&nbsp;&nbsp;|__ model.py <br/>
&nbsp;&nbsp;|__ PL_data_module.py <br/>
&nbsp;&nbsp;|__ PL_main.py <br/>
&nbsp;&nbsp;|__ PL_model.py <br/>
&nbsp;&nbsp;|__ train.py <br/>
|_ weights <br/>
&nbsp;&nbsp;|__ README.md <br/>
|_ README.md <br/>
|_ S16_PL_Transformer_En_Fr.ipynb <br/>

<h4> Details of how to run: </h4><br/>
- clone the repo with <br/>
<pre>
    <code>
        git clone git@github.com:pankaja0285/era_v1_session16.git
    </code>
</pre>
- click to open the S16_PL_Transformer_En_Fr.ipynb and proceed to run cell by cell<br/>
- The tensorboard logs are displayed realtime upon running the notebook<br/>


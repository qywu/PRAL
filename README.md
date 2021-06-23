# PRAL

Code for the paper: [A Tailored Pre-Training Model for Task-Oriented Dialog Generation](https://arxiv.org/abs/2004.13835)

## Pretrain Dataset

For the pretrain dataset, first download the repository.
```bash
# download
git clone https://github.com/qywu/DialogCorpus.git
cd DialogCorpus
```

You can manually download and process the dataset.
```bash
# download data for daily_dialog
python daily_dialog/download_data.py
# process the data
python daily_dialog/process_data.py
# the processed data is stored as the {folder_name}.json
vi daily_dialog/data/daily_dialog.json
```

Or you can just use one command.
```bash
python prepare_all_data.py \
       --download \
       --process \
       --join
```

Or you can just download our processed version:
https://drive.google.com/file/d/1VS9GndEAsrdiyIzlyhy2LAKyu_bR2Lpz/view?usp=sharing

### Detailed Dialog Processing for each dataset:

* Daily Dialog
    * Removed tokenization space for punctuations

* Persona Chat
    * Used huggingface's version [[link]](https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json)
    * Recovered lower cased utterances
    * Removed tokenization space for punctuations

* Cornell Movie Corpus
    * Ignored UTF-8 Errors
    * Extracted Names

* [Task Master](https://ai.google/tools/datasets/taskmaster-1)
    * Nothing specific

* [CCPE](https://ai.google/tools/datasets/coached-conversational-preference-elicitation)
    * Nothing specific

* [Frames](https://www.microsoft.com/en-us/research/project/frames-dataset/)
    * Nothing specific

* [Chit-Chat Challenge](https://github.com/BYU-PCCL/chitchat-dataset)
    * Nothing specific

* [Self-dialogue](https://github.com/jfainberg/self_dialogue_corpus)
    * Nothing specific

* [Schema Dialog](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
    * Nothing specific

Links


* [Daily Dialog](http://yanran.li/dailydialog) [[link]](https://github.com/qywu/DialogCorpus/tree/master/daily_dialog)

* [Conversational flow in Oxford-style debates](http://tisjune.github.io/research/iq2) [[link]](https://github.com/qywu/DialogCorpus/tree/master/debates)

* [Persona-chat](https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/convai2) [[Google Drive](https://drive.google.com/open?id=1VacuNTaQo9-tXv52XaHczPxXejRuJk9T)] 



## Training

After process or download the data, put `dialog_corpus.json` in the current directory and train the model with the following:

```
python main.py
```

## Evaluation

You can refer to ARDM's evaluation code https://github.com/qywu/ARDM. 
For the chatbot demo, you can checkout the colab example and load the pretrained weights: https://colab.research.google.com/drive/1ib7YCeNhkIDAzuOKotSlw1CfIBP_zE4r


## Pretrained Weights

We provide the download option to our pretrained weights:
https://drive.google.com/file/d/17S0TYjbUQmMzsNvfgZwY2DFULYlPQZ7h/view?usp=sharing


## Citation

You can cite the paper with:

```
@article{PRAL,
  author    = {Jing Gu and
               Qingyang Wu and
               Chongruo Wu and
               Weiyan Shi and
               Zhou Yu},
  title     = {A Tailored Pre-Training Model for Task-Oriented Dialog Generation},
  journal   = {CoRR},
  volume    = {abs/2004.13835},
  year      = {2020},
  url       = {https://arxiv.org/abs/2004.13835},
  archivePrefix = {arXiv},
  eprint    = {2004.13835},
  timestamp = {Sat, 02 May 2020 19:17:26 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2004-13835.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

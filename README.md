# ⏰ALaRM: Align Language Models via Hierarchical Rewards Modeling

<p align="left">
    <a href="https://img.shields.io/badge/PRs-Welcome-red">
        <img src="https://img.shields.io/badge/PRs-Welcome-red">
    </a>
    <a href="https://img.shields.io/github/last-commit/halfrot/ALaRM?color=green">
        <img src="https://img.shields.io/github/last-commit/halfrot/ALaRM?color=green">
    </a>
    <br/>
</p>

This repository hosts the code for the paper [ALaRM: Align Language Models via Hierarchical Rewards Modeling](https://arxiv.org/abs/2403.06754).

You can refer to our [project page](https://alarm-fdu.github.io/) for a quick project overview.

## Dependencies

```bash
# clone this repo
git clone https://github.com/halfrot/ALaRM.git
cd ALaRM
# set up conda
conda create --name env-alarm python=3.9
conda activate env-alarm
# install packages
pip install -e .
pip install -e ./trl
python -m spacy download en_core_web_sm
```

**Note: For MT task, You also need to install `java` and add it to `PATH`.**

## Usage

### Training

+ To run **Long-form QA** experiments, first download the model checkpoints from [google drive](https://drive.google.com/drive/folders/18EBBOlePyh86tsTPNeCiImKkbGqN48A7?usp=sharing), released by [FineGrainedRLHF](https://github.com/allenai/FineGrainedRLHF). And place them under `./long-form-QA/model_ckpts` folder. Then, run the following command:

```bash
accelerate launch --multi_gpu ./long-form-QA/train_ppo.py \
  --save_dir ./long-form-QA/model_ckpts/seed42/hierarchical \
  --sigmoid_shaping --reward_type hierarchical \
  --w_rel 0 --w_fact 1 --w_comp 0 --seed 42 --run_name hierarchical
```

+ To run **MT** experiments, make sure you have correctly installed `java` to run `LanguageTool`. Run the following command to start training:

```bash
accelerate launch --multi_gpu ./MT/train_ppo.py \
  --save_dir ./MT/model_ckpts/seed42/hierarchical \
  --sigmoid_shaping --reward_type hierarchical \
  --w_read 0 --w_grammar 1 --w_confidence 0 --seed 42 --run_name hierarchical
```

### Evaluation

+ To test a model on both tasks, add `--test`, specify the `--save_dir` to save result json file, and add `--policy_ckpt` to specify the model path. You may also change the test batch size by `--batch_size`. See the example command as follows:

```bash
# Long-form QA
accelerate launch --multi_gpu ./long-form-QA/train_ppo.py \
  --save_dir ./long-form-QA/model_generations/seed42/hierarchical.json \
  --sigmoid_shaping --reward_type hierarchical \
  --w_rel 0 --w_fact 1 --w_comp 0 --seed 42 --run_name test_hierarchical \
  --test --policy_ckpt ./long-form-QA/model_ckpts/t5-large-1k-train
# MT
accelerate launch --multi_gpu ./MT/train_ppo.py \
  --save_dir ./MT/model_generations/seed42/hierarchical.json \
  --sigmoid_shaping --reward_type hierarchical \
  --w_read 0 --w_grammar 1 --w_confidence 0 --seed 42 --run_name test_hierarchical \
  --test --policy_ckpt halfrot/sft-mt5-base \
  --batch_size 256
```

+ Once the result json files are saved in a directory, e.g., `./long-form-QA/model_generations/seed42`, you can get their win rates by pairwise comparison using the following command:

```bash
# without gpt-3.5-turbo
python ./eval/eval_compare.py --generations_dir ./long-form-QA/model_generations/seed42 \
  --task_type qa
# with gpt-3.5-turbo
python ./eval/eval_compare.py --generations_dir ./long-form-QA/model_generations/seed42 \
  --task_type qa --use_gpt --api_key $YOUR_OPENAI_API_KEY
# randomly select 3000 data points
python ./eval/eval_compare.py --generations_dir ./MT/model_generations/seed42 \
  --task_type mt --use_gpt --api_key $YOUR_OPENAI_API_KEY \
  --max_compare 3000
```

## Citation

If you find our work helpful, please cite as

```tex
@article{lai2024alarm,
      title={ALaRM: Align Language Models via Hierarchical Rewards Modeling}, 
      author={Lai, Yuhang and Wang, Siyuan and Liu, Shujun and Huang, Xuanjing and Wei, Zhongyu},
      journal={arXiv preprint arXiv:2403.06754},
      year={2024}
}
```

## Contributors

<a href="https://github.com/halfrot">  <img src="https://avatars.githubusercontent.com/u/58783710?s=40&v=4"  width="50" /></a> 
<a href="https://github.com/siyuanwangw">  <img src="https://avatars.githubusercontent.com/u/16791524?v=4"  width="50" /></a>
<a href="https://github.com/lsjlsj35"><img src="https://avatars.githubusercontent.com/u/103647987?v=4"  width="50" /></a>

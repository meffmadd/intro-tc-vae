import glob
import os
import pandas as pd
from pathlib import Path
from typing import List
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData


class TagConverter:
    def __init__(self, tag_path: Path, events_file:str=None) -> None:
        self.tag_path = tag_path
        if events_file is None:
            events_file = os.listdir(str(self.tag_path))[0]
        self.events_file = self.tag_path / events_file

        self.ea = event_accumulator.EventAccumulator(
            str(self.events_file),
            size_guidance={  # see below regarding this argument
                event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                event_accumulator.IMAGES: 4,
                event_accumulator.AUDIO: 4,
                event_accumulator.SCALARS: 10000,
                event_accumulator.HISTOGRAMS: 1,
                event_accumulator.TENSORS: 10,
            },
        )
        self.ea.Reload()

    @property
    def scalar_tags(self) -> List[str]:
        return self.ea.Tags()["scalars"]

    def get_df(self, tag: str) -> pd.DataFrame:
        return pd.DataFrame(self.ea.Scalars(tag))
    
    @property
    def exists(self):
        self.events_file.exists()
    
    @property
    def name(self):
        self.tag_path.name

class TensorboardReader:
    def __init__(self, run_dir: str, run: str) -> None:
        self.run_path = Path(run_dir).resolve() / run
        self._base_event = None

    @property
    def base_event(self) -> TagConverter:
        if self._base_event is None:
            base_event_file = TensorboardReader.match_name(self.run_path, "events.out.*")
            self._base_event = self.read_score("", events_file=base_event_file)
        return self._base_event

    def read_score(self, name: str, events_file:str=None) -> TagConverter:
        return TagConverter(self.run_path / name, events_file=events_file)

    @staticmethod
    def match_first(dir: str, glob_pattern: str) -> Path:
        p = Path(dir).resolve()
        return Path(glob.glob(str(p / glob_pattern))[0])
    
    @staticmethod
    def match_name(dir: str, glob_pattern: str) -> str:
        return TensorboardReader.match_first(dir=dir, glob_pattern=glob_pattern).name
    
    @property
    def exists(self):
        return self.run_path.exists()
    
    # from: https://github.com/j3soon/tbparse/blob/0a6368183b1fa3e30c4c0fd88eebb1edc10a8c5a/tbparse/summary_reader.py#L826
    @property
    def hparams(self):
        ssi_tag = "_hparams_/session_start_info"
        hparam_base_dir = self.match_name(self.run_path, "16*") # 16* because run_name is str(time.time() in SummaryWriter)
        hparam_event_score = self.read_score(hparam_base_dir, events_file=self.match_name(self.run_path / hparam_base_dir, "events.out*"))
        hparam_event_ea = hparam_event_score.ea
        hparam_content = hparam_event_ea.PluginTagToContent("hparams")
        data = hparam_content[ssi_tag]
        plugin_data: HParamsPluginData = HParamsPluginData.FromString(data)
        hparam_dict = dict(plugin_data.session_start_info.hparams)
        metric_dict = {}
        for tag in hparam_event_score.scalar_tags:
            metric_dict[tag] = hparam_event_score.get_df("lossE")["value"][0]
        return hparam_dict, metric_dict

    ### --------------
    ### SCORES 
    ### --------------

    @property
    def bvae_score(self) -> pd.DataFrame:
        return self.read_score("bvae_score_score").get_df("bvae_score")
    
    @property
    def bvae_score_scaled(self) -> pd.DataFrame:
        return self.read_score("bvae_score_scaled").get_df("bvae_score")
    
    @property
    def explicitness_score(self) -> pd.DataFrame:
        return self.read_score("mod_expl_explicitness_score").get_df("mod_expl")

    @property
    def modularity_score(self) -> pd.DataFrame:
        return self.read_score("mod_expl_modularity_score").get_df("mod_expl")
    
    @property
    def mig_score(self) -> pd.DataFrame:
        return self.base_event.get_df("mig_score")
    
    @property
    def dci_completeness_score(self) -> pd.DataFrame:
        return self.read_score("dci_dci_completeness_score").get_df("dci")
    
    @property
    def dci_disentanglement_score(self) -> pd.DataFrame:
        return self.read_score("dci_dci_disentanglement_score").get_df("dci")

    @property
    def dci_informativeness_score(self) -> pd.DataFrame:
        return self.read_score("dci_dci_informativeness_score").get_df("dci")

    ### --------------
    ### LOSSES 
    ### --------------

    @property
    def r_loss_scaled(self) -> pd.DataFrame:
        return self.read_score("losses_r_loss").get_df("losses")
    
    @property
    def r_loss(self) -> pd.DataFrame:
        return self.read_score("losses_unscaled_r_loss").get_df("losses_unscaled")

    @property
    def kl_loss_scaled(self) -> pd.DataFrame:
        return self.read_score("losses_kl").get_df("losses")
 
    @property
    def kl_loss(self) -> pd.DataFrame:
        return self.read_score("losses_unscaled_kl").get_df("losses_unscaled")

    @property
    def expelbo_f_loss_scaled(self) -> pd.DataFrame:
        return self.read_score("losses_expelbo_f").get_df("losses")

    @property
    def expelbo_f_loss(self) -> pd.DataFrame:
        return self.read_score("losses_unscaled_expelbo_f").get_df("losses_unscaled")
    
    @property
    def diff_kl(self) -> pd.DataFrame:
        return self.base_event.get_df("diff_kl")
    
    @property
    def loss_e(self) -> pd.DataFrame:
        return self.base_event.get_df("lossE")

    @property
    def loss_d(self) -> pd.DataFrame:
        return self.base_event.get_df("lossD")
    
    


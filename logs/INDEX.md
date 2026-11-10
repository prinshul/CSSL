# Logs of the reported runs

Every number of the paper's tables, the run folder it comes from and the value in that run's `results.json`. Each run folder under `runs/` holds its `config.yaml`, `train.log`, `metrics.jsonl` and `results.json`.

44 of 44 paper values match their run (to 0.005).

| Paper | Item | Run | Key in results.json | Value | Paper | Match |
|---|---|---|---|---|---|---|
| Table 1 | MoNuSeg supervised 50%, seed 0 | main_sup/monuseg_50_s0 | test_mean_dice | 76.0101 | 76.01 | yes |
| Table 1 | MoNuSeg supervised 100%, seed 0 | main_sup/monuseg_100_s0 | test_mean_dice | 77.6357 | 77.64 | yes |
| Table 1 | MoNuSeg recipe only, seed 0 | mabl_none/monuseg_50_s0 | test_mean_dice | 77.6920 | 77.69 | yes |
| Table 1 | MoNuSeg recipe only, seed 1 | mabl_none/monuseg_50_s1 | test_mean_dice | 78.4990 | 78.50 | yes |
| Table 1 | MoNuSeg recipe only, seed 2 | mabl_none/monuseg_50_s2 | test_mean_dice | 79.9854 | 79.99 | yes |
| Table 1 | MoNuSeg CSSL, seed 0 | cssl_strongest/monuseg_50_s0 | test_mean_dice | 80.6554 | 80.66 | yes |
| Table 1 | MoNuSeg CSSL, seed 1 | cssl_strongest/monuseg_50_s1 | test_mean_dice | 81.0744 | 81.07 | yes |
| Table 1 | MoNuSeg CSSL, seed 2 | cssl_strongest/monuseg_50_s2 | test_mean_dice | 80.8844 | 80.88 | yes |
| F1 calibration | supervised 50%: fg/bg F1 | main_sup/monuseg_50_s0 | selfloop_f1_fg_bg_mean | 84.4062 | 84.41 | yes |
| F1 calibration | supervised 50%: balanced accuracy | main_sup/monuseg_50_s0 | balanced_accuracy | 85.2297 | 85.23 | yes |
| F1 calibration | supervised 100%: fg/bg F1 | main_sup/monuseg_100_s0 | selfloop_f1_fg_bg_mean | 84.6154 | 84.62 | yes |
| F1 calibration | supervised 100%: balanced accuracy | main_sup/monuseg_100_s0 | balanced_accuracy | 87.4932 | 87.49 | yes |
| Appendix B | CSSL seed 0: fg/bg F1 | cssl_strongest/monuseg_50_s0 | selfloop_f1_fg_bg_mean | 87.7339 | 87.73 | yes |
| Appendix B | CSSL seed 0: balanced accuracy | cssl_strongest/monuseg_50_s0 | balanced_accuracy | 87.2700 | 87.27 | yes |
| Appendix B | CSSL seed 0: pooled-pixel nuclei F1 | cssl_strongest/monuseg_50_s0 | pooled_dice_nucleus | 81.0003 | 81.00 | yes |
| Table 2 | CHAOS supervised, split 0 | main_sup/chaos_25_s0 | test_mean_dice | 90.5525 | 90.55 | yes |
| Table 2 | CHAOS supervised, split 1 | main_sup/chaos_25_s1 | test_mean_dice | 79.0678 | 79.07 | yes |
| Table 2 | CHAOS supervised, split 2 | main_sup/chaos_25_s2 | test_mean_dice | 87.9808 | 87.98 | yes |
| Table 2 | CHAOS recipe only, split 0 | cabl_none/chaos_25_s0 | test_mean_dice | 89.9586 | 89.96 | yes |
| Table 2 | CHAOS recipe only, split 1 | cabl_none/chaos_25_s1 | test_mean_dice | 85.3325 | 85.33 | yes |
| Table 2 | CHAOS recipe only, split 2 | cabl_none/chaos_25_s2 | test_mean_dice | 91.5227 | 91.52 | yes |
| Table 2 | CHAOS CSSL, split 0 | cssl_strongest/chaos_25_s0 | test_mean_dice | 90.4679 | 90.47 | yes |
| Table 2 | CHAOS CSSL, split 1 | cssl_strongest/chaos_25_s1 | test_mean_dice | 85.6654 | 85.67 | yes |
| Table 2 | CHAOS CSSL, split 2 | cssl_strongest/chaos_25_s2 | test_mean_dice | 90.8990 | 90.90 | yes |
| Table 5 | CHAOS supervised, split 0, rounding | main_sup/chaos_25_s0 | dafnet_dice_round | 90.5976 | 90.60 | yes |
| Table 5 | CHAOS supervised, split 1, rounding | main_sup/chaos_25_s1 | dafnet_dice_round | 79.0227 | 79.02 | yes |
| Table 5 | CHAOS supervised, split 2, rounding | main_sup/chaos_25_s2 | dafnet_dice_round | 88.0254 | 88.03 | yes |
| Table 5 | CHAOS recipe only, split 0, rounding | cabl_none/chaos_25_s0 | dafnet_dice_round | 90.0176 | 90.02 | yes |
| Table 5 | CHAOS recipe only, split 1, rounding | cabl_none/chaos_25_s1 | dafnet_dice_round | 85.3153 | 85.32 | yes |
| Table 5 | CHAOS recipe only, split 2, rounding | cabl_none/chaos_25_s2 | dafnet_dice_round | 91.5403 | 91.54 | yes |
| Table 5 | CHAOS CSSL, split 0, rounding | cssl_strongest/chaos_25_s0 | dafnet_dice_round | 90.5464 | 90.55 | yes |
| Table 5 | CHAOS CSSL, split 1, rounding | cssl_strongest/chaos_25_s1 | dafnet_dice_round | 85.6648 | 85.66 | yes |
| Table 5 | CHAOS CSSL, split 2, rounding | cssl_strongest/chaos_25_s2 | dafnet_dice_round | 90.9278 | 90.93 | yes |
| Table 3 | MoNuSeg w/o contrastive loss | sabl_no_cl/monuseg_50_s0 | test_mean_dice | 78.5051 | 78.51 | yes |
| Table 3 | MoNuSeg w/o consistency loss | sabl_no_cr/monuseg_50_s0 | test_mean_dice | 80.4241 | 80.42 | yes |
| Table 3 | MoNuSeg w/o hard negatives | sabl_no_hnm/monuseg_50_s0 | test_mean_dice | 80.0659 | 80.07 | yes |
| Table 3 | MoNuSeg w/o Silhouette (fixed lambda) | sabl_fixlam/monuseg_50_s0 | test_mean_dice | 80.5407 | 80.54 | yes |
| Table 3 | MoNuSeg w/o memory bank | sabl_no_mem/monuseg_50_s0 | test_mean_dice | 81.1362 | 81.14 | yes |
| Table 3 | MoNuSeg w/o boundary class | sabl_no_bnd/monuseg_50_s0 | test_mean_dice | 77.9857 | 77.99 | yes |
| Table 3 | MoNuSeg w/o feature-level masking | sabl_no_v2/monuseg_50_s0 | test_mean_dice | 79.1075 | 79.11 | yes |
| Table 3 | CHAOS w/o contrastive loss | sabl_no_cl/chaos_25_s1 | test_mean_dice | 78.4619 | 78.46 | yes |
| Table 3 | CHAOS w/o consistency loss | sabl_no_cr/chaos_25_s1 | test_mean_dice | 86.4977 | 86.50 | yes |
| Table 3 | CHAOS w/o hard negatives | sabl_no_hnm/chaos_25_s1 | test_mean_dice | 85.1573 | 85.16 | yes |
| Table 3 | CHAOS w/o Silhouette (fixed lambda) | sabl_fixlam/chaos_25_s1 | test_mean_dice | 84.9874 | 84.99 | yes |

## Files

| File | SHA-256 |
|---|---|
| runs/cabl_none/chaos_25_s0/config.yaml | `0936913da8ad59a4ae999a5a657ca9522bcfc2a3d36b428347709d034ce5aa5d` |
| runs/cabl_none/chaos_25_s0/train.log | `9104be58036ab23b4075eab3d8499012b6352ab35bb944e74d488c6ba58f95a4` |
| runs/cabl_none/chaos_25_s0/metrics.jsonl | `f4fc3999df2019f2020b37b1c6f5c139bd092f25818d4c554caef8f9fe286191` |
| runs/cabl_none/chaos_25_s0/results.json | `5e0b8bbfe3a789d882018879c0f1f6a5578f311831034767f71b650a665ea0fe` |
| runs/cabl_none/chaos_25_s1/config.yaml | `b794ca9ab9376efab87fefe549437286d09c5152ba74e85481fb406eb64a7b4c` |
| runs/cabl_none/chaos_25_s1/train.log | `58795782161140cf17ee26a0bb825de9483dd8b7147cbf2eb834653060acbbb3` |
| runs/cabl_none/chaos_25_s1/metrics.jsonl | `d5f41d24df250e321a79038002af8b97f31ea1f7dc46780acbe3775d3305f6e8` |
| runs/cabl_none/chaos_25_s1/results.json | `9fc9b27f4dfe59b64e0f14ef12e94eef92a128ee2546f427b47e540a45b97d23` |
| runs/cabl_none/chaos_25_s2/config.yaml | `03f22fce5c924f067901ac72577fa13abc9268a3aca70824b6a0b17822d454e3` |
| runs/cabl_none/chaos_25_s2/train.log | `0fda64a68e82d6e0003d9e5abec84f3fe15b865a2f174f4a9cfed1ddd334311e` |
| runs/cabl_none/chaos_25_s2/metrics.jsonl | `444d3f181d56d75d0f6d7ede7947ee0aec8b57b27e10df619a8b0a94fe0f6d88` |
| runs/cabl_none/chaos_25_s2/results.json | `378da3d4aac4c5cdf595b832c48281330cbd9b8e26a4768603ed9349e4e6fbc7` |
| runs/cssl_strongest/chaos_25_s0/config.yaml | `cd43c2f04b62724069b9eb2c078dc639fa045cd5455cb9d5152076072ee5a99d` |
| runs/cssl_strongest/chaos_25_s0/train.log | `0fc8b8eeeff2df2c1c734127e61bb82969448f0a9679a666043fcfdc31c6cbcd` |
| runs/cssl_strongest/chaos_25_s0/metrics.jsonl | `79bf97cce0f53ba2ea0a10b1252e329038870908e88a91c053d1308a037c9f12` |
| runs/cssl_strongest/chaos_25_s0/results.json | `ae886ab4a26a65e1b64a74f107c92464fdfbe146d98d48ac1878ac7f5c1d5972` |
| runs/cssl_strongest/chaos_25_s1/config.yaml | `3f584e11072bac106d420712517320efaf908097e225cb26ad62f7087174ccd4` |
| runs/cssl_strongest/chaos_25_s1/train.log | `8bedaabd2e1129f151a240c8b991f428e83abbe31154bac1c024b80014941953` |
| runs/cssl_strongest/chaos_25_s1/metrics.jsonl | `99587066a3de2f4ec4999bccf88a6299d1ad5844474bfeed7e5a495aaa25437d` |
| runs/cssl_strongest/chaos_25_s1/results.json | `2a50f6e065e46fffce8a359c9402e2040173c74870d4f0054fd527ea2732e14b` |
| runs/cssl_strongest/chaos_25_s2/config.yaml | `19594807cf7ce706837884c6f8320c30b8b597d5d3016ea2d9cd489accb78cc9` |
| runs/cssl_strongest/chaos_25_s2/train.log | `02239ab6400d57440c57ef65d16a30a201056733ab32248e3e9f2267cfba4f7e` |
| runs/cssl_strongest/chaos_25_s2/metrics.jsonl | `8221c57dc45bd10d9640cc43e9a77b92b7c0aae0bdbe959398e1f8f90fddaa9f` |
| runs/cssl_strongest/chaos_25_s2/results.json | `b5203caac33ef61ed086ab077f3d4257d1ba20e84bd950a7dcde70b7b9dca9e1` |
| runs/cssl_strongest/monuseg_50_s0/config.yaml | `8db2ba48773672e8d269faa019d9c81fd583724c4b707d6d61598f31c80c40c5` |
| runs/cssl_strongest/monuseg_50_s0/train.log | `c2b9b358e8e0de3e294a3d51ce67fca045aafedf19bcf20779baa6924741a6b8` |
| runs/cssl_strongest/monuseg_50_s0/metrics.jsonl | `c6fff33d364e0e120c1f3df5e7c9ad754df98e0ce9cc44d4be02962aa02e76f5` |
| runs/cssl_strongest/monuseg_50_s0/results.json | `81e447c3d39ed351fed0f905d158a1f7d8595a62cffd5cf756cfce81e776d946` |
| runs/cssl_strongest/monuseg_50_s1/config.yaml | `4299cde4afa840c9969236dd84ed78c45df12ffa1e1b74611e403ebcaaba4fda` |
| runs/cssl_strongest/monuseg_50_s1/train.log | `9c042171314de8d07dea6c6ec79d75e95d56152e6cb4ccba10525b200eed5965` |
| runs/cssl_strongest/monuseg_50_s1/metrics.jsonl | `45e05da320abca7ab5b58c0115b40cf5a4eaa3d196dc6afd9d2602c7e1bd5b02` |
| runs/cssl_strongest/monuseg_50_s1/results.json | `891e9ae11a1f672a195ba205c39346170c70c6d6b1c8f88516458458282a3501` |
| runs/cssl_strongest/monuseg_50_s2/config.yaml | `03d62cab77e997c52a014acea6f50374a2bd01c9260627d5c1e4a9b54943dfaa` |
| runs/cssl_strongest/monuseg_50_s2/train.log | `03f18f625eac6e893b7acb184694a14b7e877d639a02babe6d391809e5f485f6` |
| runs/cssl_strongest/monuseg_50_s2/metrics.jsonl | `80d5dc1a4b3c58996fffedf1912ef32ead7fc3fbd23018f77c975a8ba83b025f` |
| runs/cssl_strongest/monuseg_50_s2/results.json | `356622237b46c9ee8c4df061e030f67b17e66c697b03cfefa1301cf9e66b3af6` |
| runs/mabl_none/monuseg_50_s0/config.yaml | `0ba0e630043ca4a0aa36143ec691d59d4fb8db42d18c56a8febc7f018c06144a` |
| runs/mabl_none/monuseg_50_s0/train.log | `eff4d35ed1689ae9392f1b664171ce91b71700469727e0ae84f8da352bce574a` |
| runs/mabl_none/monuseg_50_s0/metrics.jsonl | `552855fbc79e91c137c772f423635722d3629661627bad45ed0a7fceb973ab2f` |
| runs/mabl_none/monuseg_50_s0/results.json | `c0a2603f00a4b6488919aa8d8cf1e74f532beac33ddb542ba4046808685fdc36` |
| runs/mabl_none/monuseg_50_s1/config.yaml | `d9a3d314f558a4291f8768d59f079f8f1378361eab1a07a36ad48e94c5bb86dc` |
| runs/mabl_none/monuseg_50_s1/train.log | `fae8fa050b9f2babd68c839b0f0d60c2dd78e425b2f03cb0dd2269a2b84b1c43` |
| runs/mabl_none/monuseg_50_s1/metrics.jsonl | `14d59d13f419ac6ff7336f908008aaa851890812571ba5105a87110d5a1fb4a9` |
| runs/mabl_none/monuseg_50_s1/results.json | `1c6dd0b821e5c67f727e328d29dedd508fce08fbcd998ea8600107f08512385c` |
| runs/mabl_none/monuseg_50_s2/config.yaml | `1dd78ccfbafbcac7eebebb5096ea6533f2314a546824dd5681da91e7c5af4efb` |
| runs/mabl_none/monuseg_50_s2/train.log | `35be51e0b145f66c2ce27069341f4860d74a002c75098453a5318e8e00801504` |
| runs/mabl_none/monuseg_50_s2/metrics.jsonl | `6e3935ae727cd0ac47df6959b267cf47e8f50751c7d9e43a9182650d41a9417a` |
| runs/mabl_none/monuseg_50_s2/results.json | `2e8f370c3d326089b4c6b8de0ad31ba81741c52a557422e148f404f640409378` |
| runs/main_sup/chaos_25_s0/config.yaml | `8a8556c0111db15db9fc8d147787e93ca2987235e0bf1fffceb4db7da704c90c` |
| runs/main_sup/chaos_25_s0/train.log | `7720908bc40327bce73c24c5ffb5309e6e185b50d1d15c411ae42aa51d8686e5` |
| runs/main_sup/chaos_25_s0/metrics.jsonl | `69cebc6ab8e5bced6d002f88670d3a78ae6dd4a148930845ce0f17dc1b523499` |
| runs/main_sup/chaos_25_s0/results.json | `f93815aad6d7733b50b3698fd02a52159f72d5cc16748f2cebdf205b4b4e9d0b` |
| runs/main_sup/chaos_25_s1/config.yaml | `7b609d230a5a0f18c2c7d5ec2095598406405342a09b80d81d9836fb2a563888` |
| runs/main_sup/chaos_25_s1/train.log | `114b3a93d0a9bfda3d18a92c3b890c8fc8fb843ede461158b951ebe1147cc46b` |
| runs/main_sup/chaos_25_s1/metrics.jsonl | `80b496e816903ec8401b6d124fc28bfcafefd05edf5e6e6deee0231c4630fbda` |
| runs/main_sup/chaos_25_s1/results.json | `f9fd03248bbc150595f7277dbbadfb25200e2b56325bb6874c027cd086591ac2` |
| runs/main_sup/chaos_25_s2/config.yaml | `e2116cf301b0a9608a95803ab331d32e5ec4e4a72823a4b5b7ced80917179c7f` |
| runs/main_sup/chaos_25_s2/train.log | `9a30a07f3e3570046bc4cbe82baca113a7bec3eb016426020b42abea40da557e` |
| runs/main_sup/chaos_25_s2/metrics.jsonl | `7009f5f65d058a4114a19430e4578831b01c6a0e1afb8b3efa385e167f1b1145` |
| runs/main_sup/chaos_25_s2/results.json | `4d8c2b4c2a97239a667b92b96da2e5a6c8a5d95666fc4706d6c17a80e64c2d99` |
| runs/main_sup/monuseg_100_s0/config.yaml | `80d36a8ac5d3a5a4582c37086f6cd8356474c5f40e74cbec9f2de13f9600d823` |
| runs/main_sup/monuseg_100_s0/train.log | `159b5c6ce09e700875729e8db9e67d558866dd62665968fa511b06a3dbadbb2d` |
| runs/main_sup/monuseg_100_s0/metrics.jsonl | `fc90eff05a16d0d0f0db0d65add7f09233b9a3f147cd8548c65ba7f1fe07e9bf` |
| runs/main_sup/monuseg_100_s0/results.json | `23105a26f1e7fc77d88ece0ebd55e301a18ec9b2c7e957d63102fc52fe389f0c` |
| runs/main_sup/monuseg_50_s0/config.yaml | `778d01627829bb86029ee9c7bc0b71cd4807eec59bb41e9d769855e00febd001` |
| runs/main_sup/monuseg_50_s0/train.log | `d5476e72a41b72a1ae5e48e2931c87f30ee12f2d5dfe4b3fa46ca89cff9f7c9c` |
| runs/main_sup/monuseg_50_s0/metrics.jsonl | `f1fadd586f07941d24adeea0e22086c4c5bb2d3d671bc205a66c588580193485` |
| runs/main_sup/monuseg_50_s0/results.json | `c93bc6d89b67271d6ade3ec434954ff40aa0ab2fb3ad65d5aaa063f9f50fd7fa` |
| runs/sabl_fixlam/chaos_25_s1/config.yaml | `33cbcd21d11613e60303f3b587d1565c0ec2932be7465b3b649b9e2bd89e67a1` |
| runs/sabl_fixlam/chaos_25_s1/train.log | `ed008e1dc91c21387c846f73517f6db98619a751c2b1daac6b3bdcc14c2894e5` |
| runs/sabl_fixlam/chaos_25_s1/metrics.jsonl | `5e149a636a1639380dbfc29864b56321b980c8962b97240339ab42175ff8c4f7` |
| runs/sabl_fixlam/chaos_25_s1/results.json | `503b60e844e1186a0eb5d0df615c453513484a1d6540cf69a46ddb608931bf92` |
| runs/sabl_fixlam/monuseg_50_s0/config.yaml | `c7ca0361eb44e7807870718e37045337a2dbe34eef9556a83f10ecc502e42a72` |
| runs/sabl_fixlam/monuseg_50_s0/train.log | `a07982ceaad38ebebb8e947b79b33e4fc3c97751fc248d51078f1ade7785a3ea` |
| runs/sabl_fixlam/monuseg_50_s0/metrics.jsonl | `a98748aa8154f57fc7cc10207f6b9490c756eeb2970e2a39ddce85c83a929f81` |
| runs/sabl_fixlam/monuseg_50_s0/results.json | `e5634252c278a58034eec5b575ed9036a3440eaff031a5d68a9450b0f2add0e5` |
| runs/sabl_no_bnd/monuseg_50_s0/config.yaml | `f7da02177a49e5e7de34600edbc943e2b5899cf00faa70cbbeff7d35725c3d99` |
| runs/sabl_no_bnd/monuseg_50_s0/train.log | `a5ab2d8ff40520082805806f91d0aa6d0c944cda5a5b194010f8b33ae98767db` |
| runs/sabl_no_bnd/monuseg_50_s0/metrics.jsonl | `f700502da03c70d81799169eff7732d1fd0ad794f626809df64850ad2e58eb50` |
| runs/sabl_no_bnd/monuseg_50_s0/results.json | `6aec0cf51ac97fc3b9ba7416e8f8e2ef1c852db8715964a5a6fa5b9bf86500a5` |
| runs/sabl_no_cl/chaos_25_s1/config.yaml | `d4640c8d3cc4bcbb28b697d20d4c6c89d12e815a4830a17b240718fe0be4a0bf` |
| runs/sabl_no_cl/chaos_25_s1/train.log | `b07b037e03ab20d24f60c9280e0b34f7b1a85addbab93e9c67525aa8253f83d9` |
| runs/sabl_no_cl/chaos_25_s1/metrics.jsonl | `d8e3c173b41a97d5794901a4e2665f9661bb6ac82aab2434b71c1732467e1136` |
| runs/sabl_no_cl/chaos_25_s1/results.json | `7025612652a3f5dd9c5cfdd702a14cbbf380cd5485cc21da1d159bf4581b39d0` |
| runs/sabl_no_cl/monuseg_50_s0/config.yaml | `9bd19498bb0a2400ea4c78296a5586607caf122c0340f1c678af293f766cd3fe` |
| runs/sabl_no_cl/monuseg_50_s0/train.log | `b076c3dfafea39cc3c852a001f1082f99212e4f3ec955944643739d0e0e8c1dd` |
| runs/sabl_no_cl/monuseg_50_s0/metrics.jsonl | `94d6cf3bf5e1c74b1d6a072cad8c57e763685747b1fd3ad64845aa317bc9f246` |
| runs/sabl_no_cl/monuseg_50_s0/results.json | `46a3878f8cfc0139d4d2de966853e3adf56720d08314877a4367a65e249ff88b` |
| runs/sabl_no_cr/chaos_25_s1/config.yaml | `2088314b6e25e97751f1095c0cb9b5dd24a8dcd57ebca7fedcb4ef0869a69402` |
| runs/sabl_no_cr/chaos_25_s1/train.log | `6e45fbd326368e7435d352039afd6c8db38855ce1f7b80ddee506fa82f11a886` |
| runs/sabl_no_cr/chaos_25_s1/metrics.jsonl | `837a9e2a3287d7725f57d3afbf2b9958a64d3e5cc47a7af5fd6fd8aeb37a4c30` |
| runs/sabl_no_cr/chaos_25_s1/results.json | `b9b4691c9b20160f52ca18d3cdbbfd7c65887edaa7417981259b8c8fd78c98af` |
| runs/sabl_no_cr/monuseg_50_s0/config.yaml | `f00d93325cdf7e10bf845013590dbdbb00ca3b89f0ec2ea43f4d4ee2fe9c1f21` |
| runs/sabl_no_cr/monuseg_50_s0/train.log | `1f2ad24fd5a4055a0032010d2c60d37a72338675ca5c9c0c97e47b7802510829` |
| runs/sabl_no_cr/monuseg_50_s0/metrics.jsonl | `2c785caa1e233b5d9037f3b5ec534e62027a28f9fdf0fca4983f393ac72287a9` |
| runs/sabl_no_cr/monuseg_50_s0/results.json | `1e26c054d8aa6d057d8c79588e22686f5fa6f50aa1f585b4a1df675db8dcf757` |
| runs/sabl_no_hnm/chaos_25_s1/config.yaml | `089696153d84dd3a321335f996250c58e0bb63963ce0b9653e261f316d6ad40e` |
| runs/sabl_no_hnm/chaos_25_s1/train.log | `400e09287e0b3f8b5b3f46960c5ca3e600133ab765af51612327657e3d4c3407` |
| runs/sabl_no_hnm/chaos_25_s1/metrics.jsonl | `b66be564b38363c955f412601d83f0fe30a3d0c5ee6367242ee4fa8a9a22097e` |
| runs/sabl_no_hnm/chaos_25_s1/results.json | `a8b8162b1f4c6c1ec3567bfdaa7ca29fbc0a23b052bcc537953cfb9ad6879f0d` |
| runs/sabl_no_hnm/monuseg_50_s0/config.yaml | `3ccf32c46c834d42f1b03a2bab0a462c378f9bcac8a17fca56db65db3c2469b4` |
| runs/sabl_no_hnm/monuseg_50_s0/train.log | `62d15201aa2f309ec6198c76d37946b1efc8d4e95aaaaa7518afd661497ac5e9` |
| runs/sabl_no_hnm/monuseg_50_s0/metrics.jsonl | `20c113edd82483f72eb36e66b5e71ace6c0a689e27c62476e06849475c1d7326` |
| runs/sabl_no_hnm/monuseg_50_s0/results.json | `81286046ab9dbfdaa1c8d1df6c755a8292295afcae053f17d924f67d4940af39` |
| runs/sabl_no_mem/monuseg_50_s0/config.yaml | `193849d7e5c0fd577542062a3d51a3ceb5230186946176c8c79f2f3ffef85aac` |
| runs/sabl_no_mem/monuseg_50_s0/train.log | `5b5d6ea437f081cd6e4ce7568ff71187aeb251469afeb685a481ba8233b78b04` |
| runs/sabl_no_mem/monuseg_50_s0/metrics.jsonl | `84f8aeeffb707238021cb9f988f7569fe5ed26d1da6135cb3f9fbc61b5c2a3da` |
| runs/sabl_no_mem/monuseg_50_s0/results.json | `f638a16419d9d91549db0b5d03dde704ef5f63bfabb2c3fc11eddb1d134dbf45` |
| runs/sabl_no_v2/monuseg_50_s0/config.yaml | `ea047bfca0b72959ea833f7382553631e751919db45fba5a3c3d0e829400709c` |
| runs/sabl_no_v2/monuseg_50_s0/train.log | `49299da57f86307e9e34a431989348ed2e0c2fad6970226325b2fdb97f72087a` |
| runs/sabl_no_v2/monuseg_50_s0/metrics.jsonl | `28d92f30a86df7b0754b63487e4875b331210df030ed51eb07f1a0d6546de834` |
| runs/sabl_no_v2/monuseg_50_s0/results.json | `629ad9c42f0d785a8491cd8adf8a69a6cd6915f0cba8989bc347f59fc193af7d` |

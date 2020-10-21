from learning.model import MiniChessModel
from minichess.minichess import MiniChess

if __name__ == "__main__":

    meta_iterations = 10_000
    
    # OUTER LOOP

    for meta_iteration in range(meta_iterations):
        # sample across metaset (change rules)
        # rules = MiniChessRuleset(...)

        # rules -> hyperparams

        # meta learning step

        # fixed = pretrained_model
        # varied = pretrained_model (maybe with modifications!)

        # 1-shot learning step w hyperparams
        # INNER LOOP

        # eval fixed vs varied
        # did it win? what was score diff? loss?
        pass


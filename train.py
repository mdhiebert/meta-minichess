from learning.model import MiniChessModel
from minichess.minichess import MiniChess

if __name__ == "__main__":

    meta_iterations = 10_000
    
    # OUTER LOOP
    for meta_iteration in range(meta_iterations):
        # sample across metaset (change rules)
        # use rules_variance.py to generate new MiniChessRuleset(...)
        
        # execute Meta Model
        #   input: rules
        #   output: model hyperparams
        # meta learning step

        # fixed = pretrained_model -> unclear on how this is generated
        # varied = pretrained_model (maybe with modifications!)

        # 1-shot learning step w hyperparams
        # INNER LOOP

        # eval fixed vs varied
        # did it win? what was score diff? loss?
        pass


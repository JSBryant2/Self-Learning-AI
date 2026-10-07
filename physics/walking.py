"""Walking task settings and evidence-based curriculum progression."""
STAGES = (
    {'name': 'First steps', 'target_speed': .10, 'perturbation': .08},
    {'name': 'Slow walk', 'target_speed': .20, 'perturbation': .10},
    {'name': 'Steady walk', 'target_speed': .35, 'perturbation': .12},
    {'name': 'Faster walk', 'target_speed': .50, 'perturbation': .14},
)


class WalkingCurriculum:
    def __init__(self, stage=0, enabled=True):
        self.stage, self.enabled, self.streak = stage, enabled, 0
        self.transitions = []

    def consider(self, result, steps):
        target = STAGES[self.stage]['target_speed']
        def passes(row):
            return (row['upright_ratio'] >= .9 and row['forward_speed'] >= .6*target
                    and row['mean_slip_speed'] <= min(.12,target*.5) and row['body_contact_ratio'] <= .02
                    and row['moving_legs'] >= 2 and row['foot_cycles'] >= 2
                    and row.get('fall_rate',0) == 0 and row.get('reason','time_limit') == 'time_limit')
        trials = result.get('trials',[result])
        success = bool(trials) and all(passes(row) for row in trials)
        self.streak = self.streak+1 if success else 0
        if self.enabled and self.streak >= 3 and self.stage < len(STAGES)-1:
            self.transitions.append(dict(steps=steps, previous=self.stage, stage=self.stage+1))
            self.stage += 1
            self.streak = 0
            return True
        return False

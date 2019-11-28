import json
import logging
from collections import Counter, defaultdict
from hashlib import md5

import numpy as np
from tqdm.autonotebook import trange

from btb.selection.ucb1 import UCB1
from btb.tuning.tunable import Tunable
from btb.tuning.tuners.gaussian_process import GPTuner

LOGGER = logging.getLogger(__name__)


class BTBError(Exception):
    pass


class BTBSession:

    def _normalize(self, score):
        if score is not None:
            return score if self.maximization else -score

    def __init__(self, tunables, scorer, maximization=True,
                 max_errors=1, verbose=False, max_retries=10):
        self.tunables = tunables
        self.scorer = scorer
        self.maximization = maximization
        self.max_errors = max_errors

        self.best_proposal = None
        self.proposals = dict()
        self.iterations = 0
        self.errors = Counter()

        self._best_normalized = self._normalize(-np.inf)
        self._normalized_scores = defaultdict(list)
        self._tuners = dict()
        self._tunable_names = list(self.tunables.keys())
        self._selector = UCB1(self._tunable_names)

        self._range = trange if verbose else range
        self._max_retries = max_retries

    def _make_dumpable(self, to_dump):
        dumpable = dict()
        for key, value in to_dump.items():
            if not isinstance(key, str):
                key = str(key)

            if isinstance(value, np.integer):
                value = int(value)

            elif isinstance(value, np.floating):
                value = float(value)

            elif isinstance(value, np.ndarray):
                value = value.tolist()

            elif isinstance(value, np.bool_):
                value = bool(value)

            elif value == 'None':
                value = None

            dumpable[key] = value

        return dumpable

    def _make_id(self, name, config):
        dumpable_config = self._make_dumpable(config)
        proposal = {
            'name': name,
            'config': dumpable_config,
        }
        hashable = json.dumps(proposal, sort_keys=True).encode()

        return md5(hashable).hexdigest()

    def propose(self):
        if not self.tunables:
            raise BTBError('All the tunables failed.')

        if len(self._normalized_scores) < len(self._tunable_names):
            tunable_name = self._tunable_names[len(self._normalized_scores)]
            tunable_spec = self.tunables[tunable_name]

            tunable = Tunable.from_dict(tunable_spec)
            self._tuners[tunable_name] = GPTuner(tunable)
            config = self._tuners[tunable_name].propose(1)
            proposal_id = self._make_id(tunable_name, config)

        else:
            tunable_name = self._selector.select(self._normalized_scores)
            tuner = self._tuners[tunable_name]
            config = tuner.propose(1)
            proposal_id = self._make_id(tunable_name, config)

            proposals = 1
            while proposal_id in self.proposals:
                proposals += 1
                config = tuner.propose(1)
                proposal_id = self._make_id(tunable_name, config)

                if proposals >= self._max_retries:
                    LOGGER.info('No new config sampled after %s attempts. Skipping', proposals)
                    return None

        self.proposals[proposal_id] = {
            'id': proposal_id,
            'name': tunable_name,
            'config': config
        }

        return tunable_name, config

    def handle_error(self, tunable_name):
        self.errors[tunable_name] += 1
        errors = self.errors[tunable_name]

        if errors >= self.max_errors:
            LOGGER.warn('Too many errors: %s. Removing tunable %s', errors, tunable_name)
            self._normalized_scores.pop(tunable_name, None)
            self._tunable_names.remove(tunable_name)

    def record(self, tunable_name, config, score):
        proposal_id = self._make_id(tunable_name, config)
        proposal = self.proposals[proposal_id]
        proposal['score'] = score

        if score is None:
            self.handle_error(tunable_name)
        else:
            normalized = self._normalize(score)
            if score and normalized > self._best_normalized:
                LOGGER.info('New optimal found: %s - %s', tunable_name, score)
                self.best_proposal = proposal
                self._best_normalized = normalized

            try:
                tuner = self._tuners[tunable_name]
                tuner.record(config, normalized)
                self._normalized_scores[tunable_name].append(normalized)
            except Exception:
                LOGGER.exception('Could not record score to tuner')

    def run(self, iterations=10):
        for _ in self._range(iterations):
            self.iterations += 1
            proposal = self.propose()
            if proposal is None:
                continue

            tunable_name, config = proposal

            try:
                LOGGER.debug('Scoring proposal %s - %s: %s',
                             self.iterations, tunable_name, config)
                score = self.scorer(tunable_name, config)
            except Exception:
                LOGGER.exception('Proposal %s - %s crashed',
                                 self.iterations, tunable_name)
                score = None

            self.record(tunable_name, config, score)

        return self.best_proposal

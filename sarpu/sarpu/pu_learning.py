from typing import Any, Dict, Optional, Type, Union

import numpy as np
import time
from sklearn.base import BaseEstimator

from sarpu.PUmodels import BasePU, LogisticRegressionPU, PU_CLASSIFIER_REGISTRY


def _resolve_pu_model(
    model: Union[str, BaseEstimator, Type[BaseEstimator], None],
    model_kwargs: Optional[Dict[str, Any]] = None,
    default_cls: Type[BaseEstimator] = LogisticRegressionPU,
) -> BaseEstimator:
    """Instantiate or return a PU-compatible model.

    Parameters
    ----------
    model:
        Either a string key registered in :data:`PU_CLASSIFIER_REGISTRY`, a
        scikit-learn estimator instance, an estimator class, or ``None``.
    model_kwargs:
        Optional keyword arguments used when instantiating a model class.
    default_cls:
        Class used when ``model`` is ``None``.

    Returns
    -------
    BaseEstimator
        An instantiated estimator ready to be used inside the SAR-EM pipeline.
    """

    if model_kwargs is None:
        model_kwargs = {}

    if model is None:
        model_cls = default_cls
    elif isinstance(model, str):
        try:
            model_cls = PU_CLASSIFIER_REGISTRY[model]
        except KeyError as exc:
            raise ValueError(
                f"Unknown model alias '{model}'. Available options are: {', '.join(sorted(PU_CLASSIFIER_REGISTRY))}."
            ) from exc
    elif isinstance(model, type):
        model_cls = model
    elif isinstance(model, BaseEstimator):
        if model_kwargs:
            raise ValueError("'model_kwargs' cannot be provided when passing an estimator instance.")
        return model
    else:
        raise TypeError(
            "Model specification must be a string alias, estimator class, estimator instance, or None."
        )

    return model_cls(**model_kwargs)



def pu_learn_sar_e(x, s, e, classification_model=None, classification_attributes=None):
    start = time.time()
    if classification_model is None:
        classification_model = LogisticRegressionPU()
    if classification_attributes is None:
        classification_attributes = np.ones(x.shape[1]).astype(bool)

    classification_model = LimitedFeaturesModel(classification_model, classification_attributes)

    classification_model.fit(x,s,e=e)

    info = {'time':time.time()-start}

    return classification_model, info


def pu_learn_scar_c(x, s, c, classification_model=None, classification_attributes=None):
    start = time.time()
    if classification_model is None:
        classification_model = LogisticRegressionPU()
    if classification_attributes is None:
        classification_attributes = np.ones(x.shape[1]).astype(bool)

    classification_model = LimitedFeaturesModel(classification_model, classification_attributes)

    e = np.ones_like(s)*c
    classification_model.fit(x,s,e=e)

    info = {'time':time.time()-start}

    return classification_model, info



def pu_learn_neg(x, s, classification_model=None, classification_attributes=None):
    start = time.time()
    if classification_model is None:
        classification_model = LogisticRegressionPU()
    if classification_attributes is None:
        classification_attributes = np.ones(x.shape[1]).astype(bool)

    classification_model = LimitedFeaturesModel(classification_model, classification_attributes)

    e = np.ones_like(s)
    classification_model.fit(x,s)

    info = {'time':time.time()-start}

    return classification_model, info



def pu_learn_sar_em(x,
                 s,
                 propensity_attributes,
                 classification_attributes=None,
                 classification_model=None,
                 propensity_model=None,
                 max_its=500,
                 slope_eps=0.0001,
                 ll_eps=0.0001,
                 convergence_window=10,
                 refit_classifier=True
                 ):

    start = time.time()
    if classification_model is None:
        classification_model = LogisticRegressionPU()
    if propensity_model is None:
        propensity_model = LogisticRegressionPU()
    if classification_attributes is None:
        classification_attributes = np.ones(x.shape[1]).astype(bool)

    if len(propensity_attributes)==0:
        propensity_model = NoFeaturesModel()

    classification_model = LimitedFeaturesModel(classification_model, classification_attributes)
    propensity_model = LimitedFeaturesModel(propensity_model, propensity_attributes)

    info = {}

    initialize_simple(x, s, classification_model, propensity_model)

    expected_prior_y1 = classification_model.predict_proba(x)
    expected_propensity = propensity_model.predict_proba(x)
    expected_posterior_y1 = expectation_y(expected_prior_y1, expected_propensity, s)

    # loglikelihood
    ll = loglikelihood_probs(expected_prior_y1, expected_propensity, s)
    loglikelihoods = [ll]

    # propensity slope
    past_propensities = np.zeros([int(len(s)-sum(s)),convergence_window])
    propensity_slope = []
    max_ll_improvements = []

    i=0
    for i in range(max_its):
        #maximization
        propensity_model.fit(x, s, sample_weight=expected_posterior_y1)

        classification_s = np.concatenate([np.ones_like(expected_posterior_y1), np.zeros_like(expected_posterior_y1)])
        classification_weights = np.concatenate([expected_posterior_y1, 1-expected_posterior_y1])
        classification_model.fit(np.concatenate([x, x], axis=0), classification_s, sample_weight=classification_weights)


        # expectation
        expected_prior_y1 = classification_model.predict_proba(x)
        expected_propensity = propensity_model.predict_proba(x)
        expected_posterior_y1 = expectation_y(expected_prior_y1, expected_propensity, s)


        # loglikelihood
        ll = loglikelihood_probs(expected_prior_y1, expected_propensity, s)
        loglikelihoods.append(ll)

        # convergence
        push(past_propensities, expected_propensity[s==0])
        if i>convergence_window:
            max_ll_improvement = max(loglikelihoods[-convergence_window:]) - loglikelihoods[-convergence_window]
            max_ll_improvements.append(max_ll_improvement)
            average_abs_slope = np.average(np.abs(slope(past_propensities, axis=1)))
            propensity_slope.append(average_abs_slope)
            if average_abs_slope<slope_eps and max_ll_improvement < ll_eps:
                break #converged

    if refit_classifier:
        classification_model.fit(x,s,e=expected_propensity)

    info['nb_iterations']=i
    info['time']=time.time()-start
    info['loglikelihoods']=loglikelihoods
    info['propensity_slopes']=propensity_slope
    info['max_ll_improvements']=max_ll_improvements

    return classification_model, propensity_model, info


def run_sar_em_pipeline(
    x,
    s,
    propensity_attributes,
    *,
    classification_model: Union[str, BaseEstimator, Type[BaseEstimator], None] = None,
    classification_model_kwargs: Optional[Dict[str, Any]] = None,
    propensity_model: Union[str, BaseEstimator, Type[BaseEstimator], None] = None,
    propensity_model_kwargs: Optional[Dict[str, Any]] = None,
    classification_attributes=None,
    threshold_objective: Optional[float] = None,
    **sar_em_kwargs,
):
    """Convenience wrapper around :func:`pu_learn_sar_em`.

    This helper instantiates the requested models (accepting the same shorthand
    strings used in the command-line interface), forwards all keyword
    arguments to :func:`pu_learn_sar_em`, and collects the resulting artefacts
    in a single dictionary. The wrapper makes it easier to experiment with
    alternative classifiers while keeping the SAR-EM training API compact.

    Parameters
    ----------
    x, s, propensity_attributes
        See :func:`pu_learn_sar_em`.
    classification_model, propensity_model
        Model specifications accepted by :func:`_resolve_pu_model`. When
        provided as strings they must be keys of
        :data:`PU_CLASSIFIER_REGISTRY`.
    classification_model_kwargs, propensity_model_kwargs
        Optional keyword arguments passed to the instantiated models. Use these
        to, for example, set ``n_jobs=-1`` to leverage all available CPU cores.
    threshold_objective
        Optional metadata describing the thresholding objective that triggered
        the training run. The value is attached to the returned ``info``
        dictionary for downstream consumption.
    **sar_em_kwargs
        Additional keyword arguments forwarded to :func:`pu_learn_sar_em`.

    Returns
    -------
    dict
        Dictionary with the trained ``classification_model``,
        ``propensity_model`` and the ``info`` dictionary produced by
        :func:`pu_learn_sar_em`.
    """

    classification_estimator = _resolve_pu_model(
        classification_model, classification_model_kwargs, LogisticRegressionPU
    )
    propensity_estimator = _resolve_pu_model(
        propensity_model, propensity_model_kwargs, LogisticRegressionPU
    )

    trained_classifier, trained_propensity, info = pu_learn_sar_em(
        x,
        s,
        propensity_attributes,
        classification_attributes=classification_attributes,
        classification_model=classification_estimator,
        propensity_model=propensity_estimator,
        **sar_em_kwargs,
    )

    if threshold_objective is not None:
        info['threshold_objective'] = threshold_objective

    return {
        'classification_model': trained_classifier,
        'propensity_model': trained_propensity,
        'info': info,
    }


def initialize_simple(instances, labels, classification_model, propensity_model):
    """Initialization with unlabeled=negative, but reweighting the examples so that the expected class prior is 0.5"""
    proportion_labeled = labels.sum()/labels.size
    classification_weights =labels*(1-proportion_labeled)+(1-labels)*proportion_labeled
    classification_model.fit(instances,labels, sample_weight=classification_weights)
    classification_expectation = classification_model.predict_proba(instances)
    propensity_model.fit(instances,labels,sample_weight=(labels + (1-labels)*classification_expectation))

def expectation_y(expectation_f,expectation_e, s):
    result= s + (1-s) * (expectation_f*(1-expectation_e))/(1-expectation_f*expectation_e)
    return result



#  Expected loglikelihood of the model probababilities
def loglikelihood_probs(class_probabilities, propensity_scores, labels):
    prob_labeled = class_probabilities*propensity_scores
    prob_unlabeled_pos = class_probabilities*(1-propensity_scores)
    prob_unlabeled_neg = 1-class_probabilities
    prob_pos_given_unl = prob_unlabeled_pos/(prob_unlabeled_pos+prob_unlabeled_neg)
    prob_neg_given_unl = 1-prob_pos_given_unl
    return (
        labels*np.log(prob_labeled)+
        (1-labels)*(
            prob_pos_given_unl*np.log(prob_unlabeled_pos)+
            prob_neg_given_unl*np.log(prob_unlabeled_neg))
    ).mean()


def slope(array, axis=0):
    """Calculate the slope of the values in ar over dimension "axis". The values are assumed to be equidistant."""
    if axis==1:
        array = array.transpose()

    n = array.shape[0]
    norm_x = np.asarray(range(n))-(n-1)/2
    auto_cor_x = np.square(norm_x).mean(0)
    avg_y = array.mean(axis=0)
    norm_y = array - avg_y
    cov_x_y = np.matmul(norm_y.transpose(),norm_x)/n
    result = cov_x_y/auto_cor_x
    if axis==1:
        result = result.transpose()
    return result


def push(array_queue, new_array):
    array_queue[:,:-1]=array_queue[:,1:]
    array_queue[:,-1]= new_array


class NoFeaturesModel:

    def __init__(self, prior=0.5):
        self.prior = prior

    def fit(self, x,y,sample_weight=None):
        if sample_weight is None:
            self.prior=y.mean()
        else:
            self.prior = (y*sample_weight).mean()

    def predict_proba(self, x):
        return np.ones(x.shape[0])*self.prior

class LimitedFeaturesModel:
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def predict_proba(self, x):
        pr = self.model.predict_proba(x[:,self.features])
        if np.ndim(pr)>1 and np.shape(pr)[1]>1:
            pr = pr[:,1]
        return pr

    def fit(self,x,y,e=None,sample_weight=None):
        if issubclass(type(self.model), BasePU):
            self.model.fit(x[:,self.features], y,e, sample_weight)
        else:
            self.model.fit(x[:,self.features], y, sample_weight)
        return self


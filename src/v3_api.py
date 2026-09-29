"""Isolated candidate API; production api.main and models/ remain unchanged.

Run explicitly: python -m uvicorn src.v3_api:app --port 8001
LinearSVC has no native probability; its legacy confidence is an explicitly
disclosed softmax-margin proxy in this experimental service only. Promotion
would require a deliberate compatibility decision, not a silent substitution.
"""
from contextlib import asynccontextmanager
import json
import joblib
import numpy as np
from api.main import create_app
from src.calibration import uncertainty
from src.data import ROOT,sha256
from src.predict import InvalidResumeError
from src.v3_experiment import response,raw_probabilities


class CandidatePredictor:
    def __init__(self):
        folder=ROOT/'experiments/v3/models'
        manifest=json.loads((folder/'manifest.json').read_text())
        for name,digest in manifest['artifacts'].items():
            if sha256(folder/name)!=digest: raise ValueError('Candidate artifact checksum mismatch')
        self.pipeline=joblib.load(folder/'pipeline.joblib')
        self.calibrator=joblib.load(folder/'calibrator.joblib')
        self.classes=np.asarray(manifest['classes'])
        if self.pipeline.classes_.tolist()!=manifest['classes']: raise ValueError('Candidate class mismatch')
        self.policy=manifest['policy']; self.model_name='v3_'+manifest['family']

    def predict(self,text):
        if not isinstance(text,str): raise TypeError('text must be a string')
        if not text.strip() or len(text)>50000: raise InvalidResumeError('Provide nonempty text of at most 50000 characters')
        vectorizer=self.pipeline['tfidf']; model=self.pipeline['classifier']
        features=vectorizer.transform([text])
        if features.nnz==0: raise InvalidResumeError('text contains no recognized resume vocabulary')
        raw=raw_probabilities(self.pipeline,[text])[0]
        calibrated=self.calibrator.predict_proba(response(self.pipeline,[text]))[0]
        order=np.argsort(-raw,kind='stable')[:3]; cal_order=np.argsort(-calibrated,kind='stable')[:3]
        # Positive TF-IDF * coefficient for the calibrated winner's model score;
        # not a causal explanation or a decomposition of calibrated probability.
        # RF fallback retains the existing global feature-importance heuristic.
        weights=model.coef_[cal_order[0]] if hasattr(model,'coef_') else model.feature_importances_
        contributions=features.data*weights[features.indices]
        names=vectorizer.get_feature_names_out()
        terms=[str(names[features.indices[i]]) for i in np.argsort(-contributions,kind='stable') if contributions[i]>0][:8]
        words=len(text.split()); reason=uncertainty(calibrated,words,self.policy['t1'],self.policy['t2'])
        return {'predicted_category':str(self.classes[order[0]]),'confidence':float(raw[order[0]]),
            'calibrated_predicted_category':str(self.classes[cal_order[0]]),'calibrated_confidence':float(calibrated[cal_order[0]]),
            'calibrated_top_predictions':[{'category':str(self.classes[i]),'probability':float(calibrated[i])} for i in cal_order],
            'is_uncertain':reason is not None,'uncertainty_reason':reason,
            'top_predictions':[{'category':str(self.classes[i]),'probability':float(raw[i])} for i in order],
            'top_terms':terms,'text_stats':{'word_count':words,'short_input':words<50},'source':'text','text_preview':None}


def candidate_app():
    app=create_app()
    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor=CandidatePredictor()
        yield
        app.state.predictor=None
    app.router.lifespan_context=lifespan
    return app


app=candidate_app()


def capture_example():
    from fastapi.testclient import TestClient
    from src.production_v2 import save
    # Authored now, before candidate results; never a held-out record or logged CV.
    text=(
        'Backend Java developer building reliable web services and distributed applications. '
        'Implemented Java Spring Boot REST APIs with PostgreSQL and Redis, wrote SQL queries '
        'and optimized database indexes. Built authentication using OAuth and JWT, integrated '
        'Kafka messaging, and wrote JUnit and Mockito tests. Maintained Maven builds and Git '
        'workflows, containerized services with Docker, and configured CI/CD pipelines. '
        'Investigated Linux server incidents, profiled JVM memory and improved API latency. '
        'Collaborated on code reviews and documented microservice interfaces. Also worked with '
        'C++, C#, .NET, Node.js and HTTP/2 integrations.'
    )
    with TestClient(candidate_app()) as client:
        reply=client.post('/predict',json={'text':text}); reply.raise_for_status()
        body=reply.json()
    save(ROOT/'experiments/v3/api_example.json',{'status_code':reply.status_code,'response':body,
        'input_source':'Fixed authored backend/Java developer demo; not a dataset record; content not logged',
        'target_rank':next((i+1 for i,row in enumerate(body['calibrated_top_predictions']) if row['category']=='INFORMATION-TECHNOLOGY'),None)})
    print(json.dumps(body,indent=2))


if __name__=='__main__': capture_example()

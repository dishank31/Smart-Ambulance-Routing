from ..utils.config import Config

class HospitalRecommender:
    def __init__(self, df_hospitals):
        """
        Initialize with a database (DataFrame) of hospitals.
        Expects id, name, latitude, longitude at minimum.
        """
        self.df_hospitals = df_hospitals
        
    def recommend(self, severity_output, eta_predictions, bed_predictions):
        """
        Filter and rank hospitals based on inputs.
        severity_output: dict from severity model
        eta_predictions: dict mapping hospital_id to eta in mins
        bed_predictions: dict mapping hospital_id to available beds
        """
        recommendations = []
        
        recommended_dept = severity_output.get('recommended_department', 'Emergency')
        
        for idx, h in self.df_hospitals.iterrows():
            h_id = h['id']
            eta = eta_predictions.get(h_id, 999)
            beds = bed_predictions.get(h_id, 0)
            
            # Better scoring:
            # - Normalize ETA (inverse, so lower is better)
            # - Normalize beds (use a max capacity of 300 for scaling)
            
            eta_score = max(0, 45 - eta) / 45.0 # Increased range to 45 min
            bed_score = min(beds, 300) / 300.0
            
            w1 = Config.DECISION_WEIGHTS['eta_weight']
            w2 = Config.DECISION_WEIGHTS['bed_weight']
            
            # Weighted average
            score = (w1 * eta_score) + (w2 * bed_score)
            
            recommendations.append({
                'hospital_id': h_id,
                'name': h['name'],
                'eta_min': round(eta, 1),
                'beds_available': int(beds),
                'department': recommended_dept,
                'score': round(score, 3),
                'lat': h.get('latitude', 0.0),
                'lon': h.get('longitude', 0.0)
            })
            
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations

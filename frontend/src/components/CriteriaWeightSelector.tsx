import { Slider } from './ui/slider';
import { Label } from './ui/label';
import { CriteriaRequest } from '../types/types';

interface CriteriaWeightProps {
  criteriaRequest: CriteriaRequest;
  onWeightChange: (newWeights: CriteriaRequest) => void;
}

export function CriteriaWeightSelector({ criteriaRequest, onWeightChange }: CriteriaWeightProps) {
  const handleWeightChange = (type: keyof CriteriaRequest) => (value: number[]) => {
    const newWeight = value[0] / 100;
    const remainingWeight = 1 - newWeight;
    const otherTypes = Object.keys(criteriaRequest).filter(k => k !== type) as Array<keyof CriteriaRequest>;
    const sumOfOtherWeights = criteriaRequest[otherTypes[0]] + criteriaRequest[otherTypes[1]];
    const newWeights = {
      ...criteriaRequest,
      [type]: newWeight,
      [otherTypes[0]]: sumOfOtherWeights === 0 ? remainingWeight / 2 : remainingWeight * criteriaRequest[otherTypes[0]] / sumOfOtherWeights,
      [otherTypes[1]]: sumOfOtherWeights === 0 ? remainingWeight / 2 : remainingWeight * criteriaRequest[otherTypes[1]] / sumOfOtherWeights
    };
    onWeightChange(newWeights);
  };

  return (
    <div className="space-y-4">
      <div>
        <Label>Distance: {(criteriaRequest.distance * 100).toFixed(0)}%</Label>
        <Slider
          value={[criteriaRequest.distance * 100]}
          onValueChange={handleWeightChange('distance')}
          max={100}
          step={1}
        />
      </div>
      <div>
        <Label>Duration: {(criteriaRequest.duration * 100).toFixed(0)}%</Label>
        <Slider
          value={[criteriaRequest.duration * 100]}
          onValueChange={handleWeightChange('duration')}
          max={100}
          step={1}
        />
      </div>
      <div>
        <Label>Cost: {(criteriaRequest.cost * 100).toFixed(0)}%</Label>
        <Slider
          value={[criteriaRequest.cost * 100]}
          onValueChange={handleWeightChange('cost')}
          max={100}
          step={1}
        />
      </div>
    </div>
  );
}

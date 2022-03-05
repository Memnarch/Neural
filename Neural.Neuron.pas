unit Neural.Neuron;

interface

type
  TNeuron = class
  protected
    FWeights: TArray<Single>;
    FBias: Single;
  protected
    function Activation(const Value: Single): Single; virtual; abstract;
  public
    constructor Create(const ABias: Single; AWeights: TArray<Single>);
    function FeedForward(const AValues: TArray<Single>): Single;
  end;

  TNeuronClass = class of TNeuron;

  TSigmoidNeuron = class(TNeuron)
  protected
    function Activation(const Value: Single): Single; override;
  end;

implementation

{ TNeuron }

constructor TNeuron.Create(const ABias: Single; AWeights: TArray<Single>);
begin
  inherited Create();
  FBias := ABias;
  FWeights := AWeights;
end;

function TNeuron.FeedForward(const AValues: TArray<Single>): Single;
var
  i: Integer;
begin
  Result := FBias;
  for i := Low(FWeights) to High(FWeights) do
    Result := Result + (AValues[i] * FWeights[i]);
  Result := Activation(Result);
end;

{ TSigmoidNeuron }

function TSigmoidNeuron.Activation(const Value: Single): Single;
begin
  Result := 1 / (1 + Exp(-Value));
end;

end.

unit Neural.Layers;

interface

uses
  Neural;

type
  TNeuralLayer = class
  protected
    FInputShape: TShape;
    FOutputShape: TShape;
    FActivation: TActivation;
    FWeightInitializer: TInitializerFunc;
  public
    constructor Create(const AActivation: TActivation; const AWeightInitializer: TInitializerFunc); reintroduce;
    procedure Build; virtual; abstract;
    function FeedForward(const Input: TArray<Single>): TArray<Single>; virtual; abstract;
    function Backpropagade(const AGradients: TArray<Single>; const ALearningRate: Single): TArray<Single>; virtual; abstract;
    property InputShape: TShape read FInputShape write FInputShape;
    property OutputShape: TShape read FOutputShape;
  end;


implementation

{ TNeuralLayer }

constructor TNeuralLayer.Create(const AActivation: TActivation; const AWeightInitializer: TInitializerFunc);
begin
  inherited Create();
  FActivation := AActivation;
  if Assigned(AWeightInitializer) then
    FWeightInitializer := AWeightInitializer
  else
    FWeightInitializer := TInitializers.Default();
end;

end.

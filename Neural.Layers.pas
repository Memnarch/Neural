unit Neural.Layers;

interface

uses
  Neural,
  Neural.Scheduler;

type
  TNeuralLayer = class
  private
    FScheduler: IScheduler;
  protected
    FInputShape: TShape;
    FOutputShape: TShape;
    FActivation: TActivation;
    FWeightInitializer: TInitializerFunc;
    procedure RunScheduled(Count: Integer; AProc: TScheduledProc);
  public
    constructor Create(const AActivation: TActivation; const AWeightInitializer: TInitializerFunc); reintroduce;
    procedure Build; virtual; abstract;
    function FeedForward(const Input: TArray<Single>): TArray<Single>; virtual; abstract;
    function Backpropagade(const AGradients: TArray<Single>; const ALearningRate: Single): TArray<Single>; virtual; abstract;
    property InputShape: TShape read FInputShape write FInputShape;
    property OutputShape: TShape read FOutputShape;
    property Scheduler: IScheduler read FScheduler write FScheduler;
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

procedure TNeuralLayer.RunScheduled(Count: Integer; AProc: TScheduledProc);
begin
  if Assigned(FScheduler) then
    FScheduler.Run(Count, AProc)
  else
    TScheduler.Default.Run(Count, AProc);
end;

end.

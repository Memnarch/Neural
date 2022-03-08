unit Neural.Scheduler;

interface

type
  TScheduledProc = reference to procedure(Index: Integer);

  IScheduler = interface
  ['{35B0A99B-2854-4C4F-BAAA-4A55ADECF4E5}']
    procedure Run(ACount: Integer; const AProc: TScheduledProc);
  end;

  TScheduler = class(TInterfacedObject, IScheduler)
  private
    class function GetDefault: IScheduler; static;
  protected
    class var FDefault: IScheduler;
  public
    procedure Run(ACount: Integer; const AProc: TScheduledProc); virtual;
    class property Default: IScheduler read GetDefault;
  end;

implementation

{ TScheduler }

class function TScheduler.GetDefault: IScheduler;
begin
  if not Assigned(FDefault) then
    FDefault := TScheduler.Create();
  Result := FDefault;
end;

procedure TScheduler.Run(ACount: Integer; const AProc: TScheduledProc);
var
  i: Integer;
begin
  for i := 0 to Pred(ACount) do
    AProc(i);
end;

end.

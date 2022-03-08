unit Neural.Scheduler.Threaded;

interface

uses
  System.Classes,
  System.Threading,
  System.SyncObjs,
  Generics.Collections,
  Neural.Scheduler;

type
  TThreadedScheduler = class(TInterfacedObject, IScheduler)
  private
    FThreads: TObjectList<TThread>;
    FFences: TArray<THandle>;
    procedure CreateThreads(ACount: Integer);
    procedure WaitForThreads;
  public
    constructor Create(ACount: Integer);
    destructor Destroy; override;
    procedure Run(ACount: Integer; const AProc: TScheduledProc);
  end;

implementation

uses
  Windows;

type
  TSchedulerThread = class(TThread)
  private
    FCount: Integer;
    FStepSize: Integer;
    FProc: TScheduledProc;
    FOffset: Integer;
    FStart: TEvent;
    FDone: TEvent;
    function GetFence: THandle;
  protected
    procedure Execute; override;
    procedure TerminatedSet; override;
  public
    constructor Create(); reintroduce;
    destructor Destroy; override;
    procedure Trigger;
    property Proc: TScheduledProc read FProc write FProc;
    property Offset: Integer read FOffset write FOffset;
    property StepSize: Integer read FStepSize write FStepSize;
    property Count: Integer read FCount write FCount;
    property Fence: THandle read GetFence;
  end;

{ TThreadedScheduler }

constructor TThreadedScheduler.Create(ACount: Integer);
begin
  inherited Create();
  FThreads := TObjectList<TThread>.Create();
  CreateThreads(ACount);
end;

procedure TThreadedScheduler.CreateThreads(ACount: Integer);
var
  LThread: TSchedulerThread;
  i: Integer;
begin
  SetLength(FFences, ACount);
  for i := 0 to Pred(ACount) do
  begin
    LThread := TSchedulerThread.Create();
    LThread.Offset := i;
    LThread.StepSize := ACount;
    FFences[i] := LThread.Fence;
    FThreads.Add(LThread);
    LThread.Start;
  end;
end;

destructor TThreadedScheduler.Destroy;
begin
  FThreads.Free;
  inherited;
end;

procedure TThreadedScheduler.Run(ACount: Integer; const AProc: TScheduledProc);
var
  LThread: TThread;
begin
  for LThread in FThreads do
  begin
    TSchedulerThread(LThread).Proc := AProc;
    TSchedulerThread(LThread).Count := ACount;
    TSchedulerThread(LThread).Trigger;
  end;
  WaitForThreads;
end;

procedure TThreadedScheduler.WaitForThreads;
begin
  WaitForMultipleObjects(Length(FFences), @FFences[0], True, INFINITE);
end;

{ TSchedulerThread }

constructor TSchedulerThread.Create;
begin
  inherited Create(True);
  FStart := TEvent.Create(nil, False, False, '');
  FDone := TEvent.Create(Nil, False, False, '');
end;

destructor TSchedulerThread.Destroy;
begin
  inherited;
  FStart.Free;
  FDone.Free;
end;

procedure TSchedulerThread.Execute;
var
  i: Integer;
begin
  inherited;
  while not Terminated do
  begin
    FStart.WaitFor();
    if Terminated then Exit;

    i := Offset;
    while i < Count do
    begin
      FProc(i);
      Inc(i, StepSize);
    end;
    FDone.SetEvent;
  end;
end;

function TSchedulerThread.GetFence: THandle;
begin
  Result := FDone.Handle;
end;

procedure TSchedulerThread.TerminatedSet;
begin
  inherited;
  FStart.SetEvent;
end;

procedure TSchedulerThread.Trigger;
begin
  FStart.SetEvent();
end;

end.

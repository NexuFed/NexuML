// ponytail: bounded text/cursor rendering, not a PTY; use a terminal emulator if full screen/Unicode cell-width support is needed.
export class TerminalBuffer {
  private lines:string[][]=[[]];
  private row=0;
  private column=0;
  private escape="";
  private saved=[0,0];
  private trimmed=false;

  write(text:string){
    for(const char of text){
      if(this.escape){
        this.escape+=char;
        if(this.escape.startsWith("\x1b]")){
          if(char==="\x07" || this.escape.endsWith("\x1b\\") || this.escape.length>4096)this.escape="";
        }else if(this.escape.startsWith("\x1b[")){
          if(/[\x40-\x7e]/.test(char) && this.escape.length>2){this.control(this.escape);this.escape="";}
          else if(this.escape.length>128)this.escape="";
        }else if(this.escape.length===2 && char!=="[" && char!=="]")this.escape="";
        continue;
      }
      if(char==="\x1b"){this.escape=char;continue;}
      if(char==="\r"){this.column=0;continue;}
      if(char==="\n"){this.row++;this.column=0;this.ensureLine();continue;}
      if(char==="\b"){this.column=Math.max(0,this.column-1);continue;}
      if(char==="\t"){this.column=Math.min(4095,(Math.floor(this.column/8)+1)*8);continue;}
      if(char<" ")continue;
      this.ensureLine();
      const line=this.lines[this.row];
      while(line.length<this.column)line.push(" ");
      line[Math.min(this.column,4095)]=char;this.column=Math.min(4095,this.column+1);
    }
  }

  private ensureLine(){
    while(this.lines.length<=this.row)this.lines.push([]);
    if(this.lines.length>1000){const removed=this.lines.length-1000;this.lines.splice(0,removed);this.row-=removed;this.saved[0]=Math.max(0,this.saved[0]-removed);this.trimmed=true;}
  }

  private control(sequence:string){
    const command=sequence.at(-1)!;
    const params=sequence.slice(2,-1).split(";").map(value=>Number(value)||0);
    const n=Math.min(1000,Math.max(1,params[0]||1));
    if(command==="A")this.row=Math.max(0,this.row-n);
    else if(command==="B")this.row=Math.min(999,this.row+n);
    else if(command==="C")this.column=Math.min(4095,this.column+n);
    else if(command==="D")this.column=Math.max(0,this.column-n);
    else if(command==="G")this.column=Math.min(4095,n-1);
    else if(command==="H" || command==="f"){this.row=n-1;this.column=Math.min(4095,Math.max(0,(params[1]||1)-1));}
    else if(command==="s")this.saved=[this.row,this.column];
    else if(command==="u")[this.row,this.column]=this.saved;
    this.ensureLine();
    if(command==="K"){
      if(params[0]===2)this.lines[this.row]=[];
      else if(params[0]===1)this.lines[this.row].fill(" ",0,this.column+1);
      else this.lines[this.row].splice(this.column);
    }else if(command==="J"){
      if(params[0]===2){this.lines=this.lines.map(()=>[]);}
      else if(params[0]===1){for(let i=0;i<this.row;i++)this.lines[i]=[];this.lines[this.row].fill(" ",0,this.column+1);}
      else{this.lines[this.row].splice(this.column);this.lines.splice(this.row+1);}
    }
  }

  get text(){return `${this.trimmed ? "[Earlier terminal lines omitted; raw logs remain available.]\n" : ""}${this.lines.map(line=>line.join("")).join("\n")}`;}
}

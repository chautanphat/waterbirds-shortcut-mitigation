// Build the team workbook using the bundled artifact tool; JSON is pre-audited.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';

const outputDir=path.resolve('outputs/01a1ee6e-d18d-7d83-a20d-409d909f4897');
const require=createRequire(path.join(outputDir,'builder.cjs'));
const { Workbook, SpreadsheetFile }=require('@oai/artifact-tool');
const data=JSON.parse(await fs.readFile(path.join(outputDir,'comparison_data.json'),'utf8'));
const wb=Workbook.create();
const sheet=wb.worksheets.add('So sanh');
const species=wb.worksheets.add('Theo loai');
const formats={text:'@',int:'#,##0',pct:'0.00%',number:'0.00'};
const color={navy:'#294563',pale:'#F0F4F8',gray:'#5D6D80',dark:'#203047'};
const locations={};
let row=8;

function write(sheet,row,col,values){
  sheet.getRangeByIndexes(row-1,col-1,values.length,values[0].length).values=values;
}
function note(text){
  // Each row may flow into otherwise empty cells; no merged cells needed.
  const chunks=text.match(/.{1,145}(?:\s|$)/g)||[text];
  for(const chunk of chunks){write(sheet,row,1,[[chunk.trim()]]);sheet.getRange(`A${row}`).format.font={name:'Arial',size:10,color:color.gray};row++;}
}
for(const sec of data.sections){
  write(sheet,row,1,[[sec.title]]);
  sheet.getRangeByIndexes(row-1,0,1,10).format.fill=color.pale;
  sheet.getRange(`A${row}`).format.font={name:'Arial',size:12,bold:true,color:color.navy};
  sheet.getRangeByIndexes(row-1,0,1,10).format.rowHeight=28;
  row+=2;
  const header=row,bodyStart=row+1;
  write(sheet,header,1,[sec.headers]);
  write(sheet,bodyStart,1,sec.rows);
  const headerRange=sheet.getRangeByIndexes(header-1,0,1,sec.headers.length);
  headerRange.format={fill:color.navy,font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},
    wrapText:true,horizontalAlignment:'center',verticalAlignment:'center',rowHeight:42};
  for(let c=0;c<sec.headers.length;c++){
    const range=sheet.getRangeByIndexes(bodyStart-1,c,sec.rows.length,1);
    range.setNumberFormat(formats[sec.formats[c]]);
    range.format.horizontalAlignment=sec.formats[c]==='text'?'left':'right';
  }
  for(let r=0;r<sec.rows.length;r++)if(r%2)sheet.getRangeByIndexes(bodyStart+r-1,0,1,sec.headers.length).format.fill='#F5F7FA';
  locations[sec.id]={header,bodyStart,end:bodyStart+sec.rows.length-1};
  row=bodyStart+sec.rows.length+1;
  for(const n of sec.notes)note(n);
  note('Nguồn: '+sec.source);
  row+=2;
}
const protocolStart=row;
write(sheet,row,1,[['Phương pháp và giới hạn']]);row+=2;
for(const [label,text] of data.protocol){
  write(sheet,row,1,[[label]]);sheet.getRange(`A${row}`).format.font={name:'Arial',size:10,bold:true,color:color.dark};
  row++;
  note(text);row++;
}

// Base style is applied without replacing explicit number/header formatting.
sheet.showGridLines=false;sheet.tabColor=color.navy;
sheet.getRange(`A1:J${row}`).format.verticalAlignment='center';
sheet.getRange(`A1:J${row}`).format.font.name='Arial';
for(let c=0;c<10;c++)sheet.getRangeByIndexes(0,c,row,1).format.columnWidth=[27,21,24,19,19,20,23,21,23,23][c];
// Set font size only for body ranges, preserving heading sizes.
for(const sec of data.sections){
  const pos=locations[sec.id];
  sheet.getRangeByIndexes(pos.bodyStart-1,0,sec.rows.length,sec.headers.length).format.font.size=10;
  sheet.getRangeByIndexes(pos.bodyStart-1,0,sec.rows.length,sec.headers.length).format.rowHeight=25;
}
write(sheet,2,1,[[data.title]]);
sheet.getRange('A2').format.font={name:'Arial',size:16,bold:true,color:color.dark};
sheet.getRange('A2:J2').format.rowHeight=28;
sheet.getRange('A3:J3').format.borders={bottom:{style:'thin',color:color.navy}};
write(sheet,4,1,[['18/11/2026. Test gốc: 5.794 ảnh. Gaussian: 100 foreground, 20 loài. Không train lại.']]);
write(sheet,5,1,[['DPE cũng giảm mạnh Landbird khi blur. Chưa xác lập lỗi riêng của POTER hoặc khác biệt do thuật toán.']]);
write(sheet,6,1,[['ERM là enhanced ERM. POTER seed1; ERM/DPE seed0. Cặp DPE legacy thiếu hash liên kết backbone lịch sử.']]);
sheet.getRange('A4:A6').format.font={name:'Arial',size:10,color:color.gray};

// Observable arithmetic is formula-backed; source statistics and CIs stay frozen.
for(let i=0;i<3;i++){
  const clean=locations.gaussian.bodyStart+4*i;
  const strong=clean+3;
  const destination=locations.drops.bodyStart+i;
  sheet.getRange(`B${destination}:F${destination}`).formulas=[[
    `=(D${clean}-D${strong})*100`,`=(E${clean}-E${strong})*100`,
    `=(F${clean}-F${strong})*100`,`=(G${clean}-G${strong})*100`,
    `=D${destination}-E${destination}`,
  ]];
}

species.showGridLines=false;
const detail=data.species;
write(species,2,1,[['Kết quả theo từng loài và mức mờ']]);
write(species,3,1,[['5 foreground/loài, 4 nền/foreground. Không coi các composite cùng foreground là mẫu độc lập.']]);
write(species,5,1,[detail.headers]);write(species,6,1,detail.rows);
species.getRange(`A1:H${detail.rows.length+5}`).format.font={name:'Arial',size:10,color:color.dark};
species.getRange(`A1:H${detail.rows.length+5}`).format.rowHeight=23;
species.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:color.navy};
species.getRange('A3').format.font={name:'Arial',size:10,color:color.gray};
for(let c=0;c<8;c++){
  species.getRangeByIndexes(0,c,detail.rows.length+5,1).format.columnWidth=[24,17,39,17,17,17,17,19][c];
  species.getRangeByIndexes(5,c,detail.rows.length,1).setNumberFormat(formats[detail.formats[c]]);
  species.getRangeByIndexes(5,c,detail.rows.length,1).format.horizontalAlignment=detail.formats[c]==='text'?'left':'right';
}
species.tables.add(`A5:H${detail.rows.length+5}`,true,'SpeciesMetrics');
species.getRange('A5:H5').format={fill:color.navy,font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},
  wrapText:true,rowHeight:35,horizontalAlignment:'center'};
species.freezePanes.freezeRows(5);species.freezePanes.freezeColumns(3);

wb.recalculate();
const expected=data.sections.find(s=>s.id==='drops').rows;
const calculated=sheet.getRange(`B${locations.drops.bodyStart}:F${locations.drops.end}`).values;
for(let r=0;r<3;r++)for(let c=0;c<5;c++)if(Math.abs(calculated[r][c]-expected[r][c+1])>1e-8)throw new Error('Drop formula reconciliation failed');
const inspect=await wb.inspect({kind:'table',range:`'So sanh'!A${locations.drops.header}:F${locations.drops.end}`,include:'values,formulas',tableMaxRows:5,tableMaxCols:6,maxChars:3500});
console.log(inspect.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},summary:'Final error scan',maxChars:1000});
console.log(errors.ndjson);
for(const [name,range,file] of [['So sanh','A1:J34','workbook_comparison_preview.png'],['Theo loai','A1:H20','workbook_species_preview.png']]){
  const preview=await wb.render({sheetName:name,range,scale:1.25,format:'png'});
  await fs.writeFile(path.join(outputDir,file),new Uint8Array(await preview.arrayBuffer()));
}
const workbook=await SpreadsheetFile.exportXlsx(wb);
const destination=path.join(outputDir,'So_sanh_POTER_ERM_DPE_2026-11-18.xlsx');
await workbook.save(destination);
await fs.writeFile(path.join(outputDir,'workbook_validation.json'),JSON.stringify({
  drop_formulas_reconciled:true,worksheets:2,species_rows:detail.rows.length,sections:data.sections.length,
  inspect:inspect.ndjson,error_scan:errors.ndjson,locations,
},null,2));
console.log(destination);

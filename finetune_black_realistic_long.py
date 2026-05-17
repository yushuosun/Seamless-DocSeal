from __future__ import annotations
import argparse,csv,random,sys,time,copy,math
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import functional as TF
HERE=Path(__file__).parent; DOCDIFF=HERE/'DocDiff'; sys.path[:0]=[str(HERE),str(DOCDIFF)]
from DocDiff.model.DocDiff import DocDiff
from DocDiff.schedule.schedule import Schedule
from DocDiff.schedule.diffusionSample import GaussianDiffusion
from DocDiff.src.sobel import Laplacian

class PairDS(Dataset):
    def __init__(self,input_dir,gt_dir,mask_dir,meta,image_size=128,stamp_bias=.75,holdout=20,train=True):
        self.input_dir=Path(input_dir); self.gt_dir=Path(gt_dir); self.mask_dir=Path(mask_dir); self.s=image_size; self.stamp_bias=stamp_bias
        rows=list(csv.DictReader(open(meta,encoding='utf-8-sig'))); files=sorted({r['file'] for r in rows})
        hold=set(files[-holdout:]) if holdout>0 else set()
        self.rows=[r for r in rows if ((r['file'] not in hold) if train else (r['file'] in hold)) and (self.input_dir/r['file']).exists() and (self.gt_dir/r['file']).exists()]
        self.by_file={}
        for r in self.rows: self.by_file.setdefault(r['file'],[]).append(r)
        self.files=sorted(self.by_file)
        print(('[train]' if train else '[holdout]'),len(self.files),'files',len(self.rows),'stamp rows')
    def __len__(self): return max(1,len(self.rows))
    def _crop_box(self,W,H,r):
        s=self.s
        if random.random()<self.stamp_bias:
            x=int(float(r.get('x',0))); y=int(float(r.get('y',0))); w=int(float(r.get('stamp_w',s))); h=int(float(r.get('stamp_h',s)))
            cx=x+w//2+random.randint(-max(1,w//3),max(1,w//3)); cy=y+h//2+random.randint(-max(1,h//3),max(1,h//3))
            x1=max(0,min(max(0,W-s),cx-s//2)); y1=max(0,min(max(0,H-s),cy-s//2))
        else:
            x1=random.randint(0,max(0,W-s)); y1=random.randint(0,max(0,H-s))
        return x1,y1,x1+s,y1+s
    def __getitem__(self,idx):
        r=self.rows[idx%len(self.rows)]; f=r['file']
        inp=Image.open(self.input_dir/f).convert('RGB'); gt=Image.open(self.gt_dir/f).convert('RGB')
        m=Image.open(self.mask_dir/f).convert('L') if (self.mask_dir/f).exists() else Image.new('L',inp.size,0)
        W,H=inp.size; box=self._crop_box(W,H,r); inp=inp.crop(box); gt=gt.crop(box); m=m.crop(box)
        if random.random()<.5: inp=TF.hflip(inp); gt=TF.hflip(gt); m=TF.hflip(m)
        ang=random.uniform(-6,6)
        if abs(ang)>.5:
            inp=TF.rotate(inp,ang,fill=255); gt=TF.rotate(gt,ang,fill=255); m=TF.rotate(m,ang,fill=0)
        return TF.to_tensor(inp),TF.to_tensor(gt),TF.to_tensor(m)
    def fixed_holdout_items(self,max_items=64):
        items=[]
        for f in self.files[:max_items]:
            r=self.by_file[f][0]; inp=Image.open(self.input_dir/f).convert('RGB'); gt=Image.open(self.gt_dir/f).convert('RGB'); m=Image.open(self.mask_dir/f).convert('L') if (self.mask_dir/f).exists() else Image.new('L',inp.size,0)
            W,H=inp.size; x=int(float(r.get('x',0))); y=int(float(r.get('y',0))); w=int(float(r.get('stamp_w',self.s))); h=int(float(r.get('stamp_h',self.s))); cx=x+w//2; cy=y+h//2; x1=max(0,min(max(0,W-self.s),cx-self.s//2)); y1=max(0,min(max(0,H-self.s),cy-self.s//2)); box=(x1,y1,x1+self.s,y1+self.s)
            items.append((TF.to_tensor(inp.crop(box)),TF.to_tensor(gt.crop(box)),TF.to_tensor(m.crop(box)),f))
        return items

def cfg():
    return type('Cfg',(),{'IMAGE_SIZE':[128,128],'CHANNEL_X':3,'CHANNEL_Y':3,'MODEL_CHANNELS':32,'NUM_RESBLOCKS':1,'CHANNEL_MULT':[1,2,3,4],'NUM_HEADS':1,'TIMESTEPS':100,'SCHEDULE':'linear','PRE_ORI':'True','BETA_LOSS':50,'HIGH_LOW_FREQ':'True'})()
def make_net(c,device):
    return DocDiff(input_channels=c.CHANNEL_X+c.CHANNEL_Y,output_channels=c.CHANNEL_Y,n_channels=c.MODEL_CHANNELS,ch_mults=c.CHANNEL_MULT,n_blocks=c.NUM_RESBLOCKS).to(device)
def psnr(a,b):
    mse=torch.mean((a-b)**2).item(); return 99.0 if mse<=1e-12 else 10*math.log10(1.0/mse)
def weighted_mse(a,b,m,stamp_weight):
    w=1+(stamp_weight-1)*m.clamp(0,1); return torch.mean(w*(a-b)**2)
def update_ema(ema,net,decay):
    with torch.no_grad():
        for ep,p in zip(ema.parameters(),net.parameters()): ep.data.mul_(decay).add_(p.data,alpha=1-decay)
def eval_init(net,items,device,stamp_weight):
    net.eval(); vals=[]; vals_stamp=[]
    with torch.no_grad():
        for inp,gt,m,_ in items:
            inp=inp.unsqueeze(0).to(device); gt=gt.unsqueeze(0).to(device); m=m.unsqueeze(0).to(device)
            pred=net.init_predictor(inp,torch.zeros((1,),device=device,dtype=torch.long)).clamp(0,1)
            vals.append(psnr(pred,gt))
            if m.sum()>1: vals_stamp.append(psnr(pred*m,gt*m))
    return float(np.mean(vals)),float(np.mean(vals_stamp)) if vals_stamp else 0.0

def main():
    ap=argparse.ArgumentParser()
    root=r'E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_realistic_scan_3000'
    ap.add_argument('--input_dir',default=root+r'\input'); ap.add_argument('--gt_dir',default=root+r'\gt'); ap.add_argument('--mask_dir',default=root+r'\stamp_mask'); ap.add_argument('--meta',default=root+r'\meta.csv')
    ap.add_argument('--init_w',default=str(DOCDIFF/'checksave'/'seal_init_black.pth')); ap.add_argument('--den_w',default=str(DOCDIFF/'checksave'/'seal_denoiser_black.pth'))
    ap.add_argument('--out_init',default=str(DOCDIFF/'checksave'/'seal_init_black_realistic_long.pth')); ap.add_argument('--out_den',default=str(DOCDIFF/'checksave'/'seal_denoiser_black_realistic_long.pth'))
    ap.add_argument('--iters',type=int,default=50000); ap.add_argument('--batch',type=int,default=8); ap.add_argument('--lr',type=float,default=1.5e-5); ap.add_argument('--image_size',type=int,default=128); ap.add_argument('--holdout',type=int,default=20)
    ap.add_argument('--ema_decay',type=float,default=.9995); ap.add_argument('--stamp_weight',type=float,default=5.0); ap.add_argument('--stamp_bias',type=float,default=.80); ap.add_argument('--num_workers',type=int,default=2); ap.add_argument('--save_every',type=int,default=5000); ap.add_argument('--eval_every',type=int,default=1000); ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('[device]',device,torch.__version__); print('[gpu]',torch.cuda.get_device_name(0) if device.type=='cuda' else '')
    c=cfg(); c.IMAGE_SIZE=[args.image_size,args.image_size]; net=make_net(c,device); net.init_predictor.load_state_dict(torch.load(args.init_w,map_location=device)); net.denoiser.load_state_dict(torch.load(args.den_w,map_location=device))
    ema=copy.deepcopy(net).to(device).eval(); sched=Schedule(c.SCHEDULE,c.TIMESTEPS); diff=GaussianDiffusion(net.denoiser,c.TIMESTEPS,sched).to(device); hf=Laplacian().to(device); opt=optim.AdamW(net.parameters(),lr=args.lr,weight_decay=1e-4)
    train=PairDS(args.input_dir,args.gt_dir,args.mask_dir,args.meta,args.image_size,args.stamp_bias,args.holdout,True); hold=PairDS(args.input_dir,args.gt_dir,args.mask_dir,args.meta,args.image_size,args.stamp_bias,args.holdout,False); hold_items=hold.fixed_holdout_items(64)
    dl=DataLoader(train,batch_size=args.batch,shuffle=True,drop_last=True,num_workers=args.num_workers,pin_memory=(device.type=='cuda'))
    Path(args.out_init).parent.mkdir(parents=True,exist_ok=True); log=Path(args.out_init).with_suffix('.train_log.csv'); lf=open(log,'w',newline='',encoding='utf-8-sig'); wr=csv.DictWriter(lf,fieldnames=['iter','loss','ddpm','pix','hold_psnr','hold_stamp_psnr','ema_psnr','ema_stamp_psnr','sec']); wr.writeheader()
    t0=time.time(); it=0; losses=[]; print('[train] iters',args.iters,'batch',args.batch,'lr',args.lr,'img',args.image_size,'stamp_weight',args.stamp_weight,'holdout',args.holdout)
    while it<args.iters:
        for img,gt,m in dl:
            if it>=args.iters: break
            img=img.to(device,non_blocking=True); gt=gt.to(device,non_blocking=True); m=m.to(device,non_blocking=True); net.train(); opt.zero_grad(set_to_none=True)
            t=torch.randint(0,c.TIMESTEPS,(img.shape[0],),device=device).long(); init_pred,noise_pred,_,_=net(gt,img,t,diff)
            residual=gt-init_pred; ddpm=2*weighted_mse(hf(noise_pred),hf(residual),m,args.stamp_weight)+weighted_mse(noise_pred,residual,m,args.stamp_weight)
            pix=weighted_mse(init_pred,gt,m,args.stamp_weight)+2*weighted_mse(init_pred-hf(init_pred),gt-hf(gt),m,args.stamp_weight)
            loss=ddpm+c.BETA_LOSS*pix/c.TIMESTEPS; loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); update_ema(ema,net,args.ema_decay)
            losses.append(loss.item())
            if it%50==0:
                ips=(it+1)/max(1e-6,time.time()-t0); print(f'iter {it:6d}/{args.iters} loss={np.mean(losses[-50:]):.5f} ddpm={ddpm.item():.5f} pix={pix.item():.5f} {ips:.2f} it/s ETA={(args.iters-it)/max(ips,1e-6)/3600:.2f}h')
            if it%args.eval_every==0:
                hp,hsp=eval_init(net,hold_items,device,args.stamp_weight); ep,esp=eval_init(ema,hold_items,device,args.stamp_weight); wr.writerow({'iter':it,'loss':np.mean(losses[-100:]),'ddpm':ddpm.item(),'pix':pix.item(),'hold_psnr':hp,'hold_stamp_psnr':hsp,'ema_psnr':ep,'ema_stamp_psnr':esp,'sec':time.time()-t0}); lf.flush(); print(f'[eval] iter={it} hold={hp:.3f}/{hsp:.3f} ema={ep:.3f}/{esp:.3f}')
            it+=1
            if it%args.save_every==0:
                torch.save(net.init_predictor.state_dict(),args.out_init); torch.save(net.denoiser.state_dict(),args.out_den); torch.save(ema.init_predictor.state_dict(),Path(args.out_init).with_name(Path(args.out_init).stem+'_ema.pth')); torch.save(ema.denoiser.state_dict(),Path(args.out_den).with_name(Path(args.out_den).stem+'_ema.pth')); print('[save]',it)
    torch.save(net.init_predictor.state_dict(),args.out_init); torch.save(net.denoiser.state_dict(),args.out_den); torch.save(ema.init_predictor.state_dict(),Path(args.out_init).with_name(Path(args.out_init).stem+'_ema.pth')); torch.save(ema.denoiser.state_dict(),Path(args.out_den).with_name(Path(args.out_den).stem+'_ema.pth')); lf.close(); print('[done]',time.time()-t0,'log',log)
if __name__=='__main__': main()

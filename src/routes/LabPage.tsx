import { useParams } from 'react-router-dom'
import { AttentionLab } from '../components/labs/AttentionLab'
import { KvCacheLab } from '../components/labs/KvCacheLab'
import { SamplingLab } from '../components/labs/SamplingLab'
import { NotFoundPage } from './NotFoundPage'

export function LabPage() {
  const { labId } = useParams()
  if (labId === 'attention') return <AttentionLab />
  if (labId === 'kv-cache') return <KvCacheLab />
  if (labId === 'sampling') return <SamplingLab />
  return <NotFoundPage />
}
